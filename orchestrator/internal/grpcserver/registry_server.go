// Package grpcserver adapts the orchestrator's internal registry
// (internal/registry) to the generated WorkerRegistry gRPC service
// (internal/proto/registry), translating between proto messages and
// plain domain structs so the registry package itself has no
// dependency on protobuf.
package grpcserver

import (
	"context"
	"errors"
	"log"
	"time"

	commonpb "orchestrator/internal/proto/common"
	registrypb "orchestrator/internal/proto/registry"
	"orchestrator/internal/registry"
)

// Options configures policy that lives at the RPC boundary rather than
// in storage.
type Options struct {
	// AllowSimulatedWorkers: accept registrations whose hardware
	// profile is PROFILE_SOURCE_SIMULATED. Off by default so a
	// production cluster can never silently mix hypothetical hardware
	// into its view of real machines.
	AllowSimulatedWorkers bool
}

// RegistryServer implements registrypb.WorkerRegistryServer on top of a
// registry.Registry.
type RegistryServer struct {
	registrypb.UnimplementedWorkerRegistryServer
	store registry.Registry
	opts  Options
}

// NewRegistryServer wraps store as a gRPC WorkerRegistry service with
// default Options (simulated workers rejected).
func NewRegistryServer(store registry.Registry) *RegistryServer {
	return NewRegistryServerWithOptions(store, Options{})
}

// NewRegistryServerWithOptions is NewRegistryServer with explicit policy.
func NewRegistryServerWithOptions(store registry.Registry, opts Options) *RegistryServer {
	return &RegistryServer{store: store, opts: opts}
}

func toProfileSource(s commonpb.ProfileSource) registry.ProfileSource {
	switch s {
	case commonpb.ProfileSource_PROFILE_SOURCE_REAL:
		return registry.ProfileSourceReal
	case commonpb.ProfileSource_PROFILE_SOURCE_SIMULATED:
		return registry.ProfileSourceSimulated
	default:
		return registry.ProfileSourceUnknown
	}
}

func unixMs(ms int64) time.Time {
	if ms <= 0 {
		return time.Time{}
	}
	return time.UnixMilli(ms)
}

func (s *RegistryServer) Register(ctx context.Context, req *registrypb.RegisterRequest) (*registrypb.RegisterResponse, error) {
	if req.GetWorkerId() == "" {
		return &registrypb.RegisterResponse{
			Accepted: false,
			Message:  "worker_id is required",
		}, nil
	}

	hw := req.GetHardware()
	rec := registry.WorkerRecord{
		ID:              req.GetWorkerId(),
		Address:         req.GetAddress(),
		Hostname:        hw.GetHostname(),
		CPULogicalCores: hw.GetCpuLogicalCores(),
		TotalRAMBytes:   hw.GetTotalRamBytes(),
		HasCUDAGPU:      hw.GetHasCudaGpu(),
		GPUName:         hw.GetGpuName(),
		GPUVRAMBytes:    hw.GetGpuVramBytes(),
		OS:              hw.GetOs(),
		PythonVersion:   hw.GetPythonVersion(),
		Status:          "idle",

		// --- Phase 3 ---
		HardwareSchemaVersion: hw.GetHardwareSchemaVersion(),
		CPUManufacturer:       hw.GetCpuManufacturer(),
		CPUModel:              hw.GetCpuModel(),
		CPUPhysicalCores:      hw.GetCpuPhysicalCores(),
		CPUArchitecture:       hw.GetCpuArchitecture(),
		OSName:                hw.GetOsName(),
		OSVersion:             hw.GetOsVersion(),
		OSBuild:               hw.GetOsBuild(),
		GPUs:                  toRegistryGPUs(hw.GetGpus()),
		TorchInstalled:        hw.GetTorchInstalled(),
		TorchVersion:          hw.GetTorchVersion(),
		CPUBackendAvailable:   hw.GetCpuBackendAvailable(),

		// --- schema 3 ---
		ProfileSource:         toProfileSource(hw.GetProfileSource()),
		ProfileCollectedAt:    unixMs(hw.GetProfileCollectedAtUnixMs()),
		SimulatedProfileName:  hw.GetSimulatedProfileName(),
		BackendCapabilities:   toRegistryBackends(hw.GetBackendCapabilities()),
		TorchCUDABuildVersion: hw.GetTorchCudaBuildVersion(),
		GPUMetrics:            toRegistryGPUMetrics(hw.GetGpuMetrics()),
		BackendProbePending:   hw.GetBackendProbePending(),
	}

	if rec.ProfileSource == registry.ProfileSourceSimulated && !s.opts.AllowSimulatedWorkers {
		log.Printf("rejected registration of %q: SIMULATED hardware profile %q while allow_simulated_workers is false",
			rec.ID, rec.SimulatedProfileName)
		return &registrypb.RegisterResponse{
			Accepted: false,
			Message: "this orchestrator does not accept SIMULATED hardware profiles " +
				"(set allow_simulated_workers: true in orchestrator.yaml for development/testing)",
		}, nil
	}

	// hw itself may be nil (a RegisterRequest with no hardware field
	// set at all, e.g. some older tests) -- the Get* accessors above
	// are nil-safe generated methods, but these are raw pointer fields
	// (needed to preserve the tri-state nil-vs-zero distinction), so
	// they need an explicit nil guard rather than relying on that.
	if hw != nil {
		rec.CUDAAvailable = hw.CudaAvailable
		rec.CUDADeviceCount = hw.CudaDeviceCount
		rec.DirectMLAvailable = hw.DirectmlAvailable
		rec.CPUUtilizationPercent = hw.CpuUtilizationPercent
		rec.RAMAvailableBytes = hw.RamAvailableBytes
		rec.RAMUsedBytes = hw.RamUsedBytes
		rec.RAMUtilizationPercent = hw.RamUtilizationPercent
		rec.CPUFrequencyCurrentMHz = hw.CpuFrequencyCurrentMhz
		rec.CPUFrequencyMaxMHz = hw.CpuFrequencyMaxMhz
		rec.CPULoadAverage1m = hw.CpuLoadAverage_1M
		rec.SwapTotalBytes = hw.SwapTotalBytes
		rec.SwapUsedBytes = hw.SwapUsedBytes
		rec.SwapUtilizationPercent = hw.SwapUtilizationPercent
	}

	isNew, err := s.store.Register(rec)
	if err != nil {
		if errors.Is(err, registry.ErrInvalidHardware) {
			log.Printf("rejected registration of %q: %v", rec.ID, err)
		}
		return &registrypb.RegisterResponse{
			Accepted: false,
			Message:  err.Error(),
		}, nil
	}

	msg := "re-registered"
	if isNew {
		msg = "registered"
	}
	return &registrypb.RegisterResponse{
		Accepted:         true,
		Message:          msg,
		ServerTimeUnixMs: time.Now().UnixMilli(),
	}, nil
}

func (s *RegistryServer) Heartbeat(ctx context.Context, req *registrypb.HeartbeatRequest) (*registrypb.HeartbeatResponse, error) {
	metrics := registry.LiveMetrics{
		CPUUtilizationPercent: req.CpuUtilizationPercent,
		RAMAvailableBytes:     req.RamAvailableBytes,
		RAMUsedBytes:          req.RamUsedBytes,
		RAMUtilizationPercent: req.RamUtilizationPercent,

		CPUFrequencyCurrentMHz: req.CpuFrequencyCurrentMhz,
		CPULoadAverage1m:       req.CpuLoadAverage_1M,
		SwapTotalBytes:         req.SwapTotalBytes,
		SwapUsedBytes:          req.SwapUsedBytes,
		SwapUtilizationPercent: req.SwapUtilizationPercent,
		GPUMetrics:             toRegistryGPUMetrics(req.GetGpuMetrics()),
		CollectedAt:            unixMs(req.GetMetricsCollectedAtUnixMs()),
	}
	err := s.store.HeartbeatWithMetrics(req.GetWorkerId(), req.GetStatus(), time.Now(), metrics)
	resp := &registrypb.HeartbeatResponse{
		Acknowledged:     err == nil,
		ServerTimeUnixMs: time.Now().UnixMilli(),
	}
	if err != nil {
		resp.Message = err.Error()
		resp.ReregisterRequired = errors.Is(err, registry.ErrWorkerNotFound)
	}
	return resp, nil
}

func (s *RegistryServer) Deregister(ctx context.Context, req *registrypb.DeregisterRequest) (*registrypb.DeregisterResponse, error) {
	err := s.store.Deregister(req.GetWorkerId())
	return &registrypb.DeregisterResponse{
		Accepted: err == nil,
	}, nil
}

// toRegistryGPUs converts the proto GPU list to the registry package's
// plain domain type, so that package stays free of a protobuf
// dependency. nil-safe: a nil/empty input yields a nil slice.
func toRegistryGPUs(src []*commonpb.GPUInfo) []registry.GPUInfo {
	if len(src) == 0 {
		return nil
	}
	out := make([]registry.GPUInfo, 0, len(src))
	for _, g := range src {
		if g == nil {
			continue
		}
		out = append(out, registry.GPUInfo{
			Vendor:           g.GetVendor(),
			Name:             g.GetName(),
			MemoryTotalBytes: g.MemoryTotalBytes,
			MemoryType:       g.GetMemoryType(),
			Backends:         g.GetBackends(),
			DriverVersion:    g.GetDriverVersion(),

			ID:                   g.GetId(),
			DedicatedMemoryBytes: g.DedicatedMemoryBytes,
			SharedMemoryBytes:    g.SharedMemoryBytes,
			MemorySource:         g.GetMemorySource(),
			Integrated:           g.GetIntegrated(),
		})
	}
	return out
}

func toRegistryGPUMetrics(src []*commonpb.GPUMetrics) []registry.GPUMetrics {
	if len(src) == 0 {
		return nil
	}
	out := make([]registry.GPUMetrics, 0, len(src))
	for _, m := range src {
		if m == nil {
			continue
		}
		out = append(out, registry.GPUMetrics{
			GPUID:                    m.GetGpuId(),
			UtilizationPercent:       m.UtilizationPercent,
			DedicatedMemoryUsedBytes: m.DedicatedMemoryUsedBytes,
			SharedMemoryUsedBytes:    m.SharedMemoryUsedBytes,
		})
	}
	return out
}

func toRegistryBackends(src []*commonpb.BackendCapability) []registry.BackendCapability {
	if len(src) == 0 {
		return nil
	}
	out := make([]registry.BackendCapability, 0, len(src))
	for _, b := range src {
		if b == nil {
			continue
		}
		out = append(out, registry.BackendCapability{
			Name:        b.GetName(),
			Available:   b.Available,
			DeviceCount: b.GetDeviceCount(),
			Version:     b.GetVersion(),
			Devices:     b.GetDevices(),
			Tested:      b.GetTested(),
			TestPassed:  b.TestPassed,
			TestGFLOPS:  b.TestGflops,
			Detail:      b.GetDetail(),
		})
	}
	return out
}
