package grpcserver

import (
	"context"
	"strings"
	"testing"

	commonpb "orchestrator/internal/proto/common"
	registrypb "orchestrator/internal/proto/registry"
	"orchestrator/internal/registry"
)

func p64(v int64) *int64      { return &v }
func pf64(v float64) *float64 { return &v }
func pbool(v bool) *bool      { return &v }

func schema3Hardware(source commonpb.ProfileSource, simName string) *commonpb.HardwareInfo {
	return &commonpb.HardwareInfo{
		Hostname:                 "host",
		CpuLogicalCores:          12,
		CpuPhysicalCores:         10,
		TotalRamBytes:            16 << 30,
		HardwareSchemaVersion:    3,
		ProfileSource:            source,
		SimulatedProfileName:     simName,
		ProfileCollectedAtUnixMs: 1793407905533,
		CpuFrequencyMaxMhz:       pf64(1700),
		SwapTotalBytes:           p64(14 << 30),
		TorchCudaBuildVersion:    "12.6",
		Gpus: []*commonpb.GPUInfo{{
			Id: "luid:A", Vendor: "Intel", Name: "Intel(R) Iris(R) Xe Graphics", MemoryType: "shared",
			Integrated: true, DedicatedMemoryBytes: p64(134217728), SharedMemoryBytes: p64(8438444032), MemorySource: "dxgi",
		}},
		BackendCapabilities: []*commonpb.BackendCapability{
			{Name: "cpu", Available: pbool(true), Tested: true, TestPassed: pbool(true), TestGflops: pf64(155.89), DeviceCount: 1},
			{Name: "cuda", Available: pbool(false), Version: "12.6", Detail: "no NVIDIA GPU"},
		},
		GpuMetrics: []*commonpb.GPUMetrics{{GpuId: "luid:A", SharedMemoryUsedBytes: p64(1 << 30)}},
	}
}

func TestRegister_Schema3FieldsRoundTrip(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	srv := NewRegistryServer(store)
	resp, err := srv.Register(context.Background(), &registrypb.RegisterRequest{
		WorkerId: "w", Address: "127.0.0.1:1", Hardware: schema3Hardware(commonpb.ProfileSource_PROFILE_SOURCE_REAL, ""),
	})
	if err != nil || !resp.Accepted {
		t.Fatalf("Register: %v %+v", err, resp)
	}
	rec, _ := store.Get("w")
	if rec.ProfileSource != registry.ProfileSourceReal || rec.ProfileCollectedAt.UnixMilli() != 1793407905533 {
		t.Errorf("source/timestamp: %q %v", rec.ProfileSource, rec.ProfileCollectedAt)
	}
	g := rec.GPUs[0]
	if g.ID != "luid:A" || !g.Integrated || *g.SharedMemoryBytes != 8438444032 || g.MemoryTotalBytes != nil {
		t.Errorf("gpu: %+v", g)
	}
	if len(rec.BackendCapabilities) != 2 || *rec.BackendCapabilities[0].TestGFLOPS != 155.89 || rec.BackendCapabilities[1].Usable() {
		t.Errorf("backends: %+v", rec.BackendCapabilities)
	}
	if rec.TorchCUDABuildVersion != "12.6" || *rec.SwapTotalBytes != 14<<30 || *rec.CPUFrequencyMaxMHz != 1700 {
		t.Error("scalar schema-3 fields lost")
	}
	if len(rec.GPUMetrics) != 1 || *rec.GPUMetrics[0].SharedMemoryUsedBytes != 1<<30 {
		t.Errorf("registration gpu metrics: %+v", rec.GPUMetrics)
	}
	if rec.CPULoadAverage1m != nil {
		t.Error("unset optional must stay nil (unknown)")
	}
	if rec.BackendProbePending {
		t.Error("probe not pending in this fixture")
	}

	hw := schema3Hardware(commonpb.ProfileSource_PROFILE_SOURCE_REAL, "")
	hw.BackendProbePending = true
	hw.BackendCapabilities = nil
	if resp, _ := srv.Register(context.Background(), &registrypb.RegisterRequest{WorkerId: "w", Address: "127.0.0.1:1", Hardware: hw}); !resp.Accepted {
		t.Fatal(resp.Message)
	}
	if rec, _ := store.Get("w"); !rec.BackendProbePending {
		t.Error("backend_probe_pending not stored")
	}
}

func TestRegister_SimulatedRejectedByDefault(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	srv := NewRegistryServer(store)
	resp, _ := srv.Register(context.Background(), &registrypb.RegisterRequest{
		WorkerId: "sim", Address: "a", Hardware: schema3Hardware(commonpb.ProfileSource_PROFILE_SOURCE_SIMULATED, "profile-b"),
	})
	if resp.Accepted || !strings.Contains(resp.Message, "SIMULATED") {
		t.Fatalf("simulated registration should be rejected by default: %+v", resp)
	}
	if _, ok := store.Get("sim"); ok {
		t.Fatal("rejected simulated worker was stored")
	}
}

func TestRegister_SimulatedAcceptedWhenAllowedAndLabelled(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	srv := NewRegistryServerWithOptions(store, Options{AllowSimulatedWorkers: true})
	resp, _ := srv.Register(context.Background(), &registrypb.RegisterRequest{
		WorkerId: "sim", Address: "a", Hardware: schema3Hardware(commonpb.ProfileSource_PROFILE_SOURCE_SIMULATED, "profile-b"),
	})
	if !resp.Accepted {
		t.Fatalf("rejected: %s", resp.Message)
	}
	rec, _ := store.Get("sim")
	if rec.ProfileSource != registry.ProfileSourceSimulated || rec.SimulatedProfileName != "profile-b" {
		t.Errorf("stored as %q/%q", rec.ProfileSource, rec.SimulatedProfileName)
	}
}

func TestRegister_InvalidHardwareRejected(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	srv := NewRegistryServer(store)
	hw := schema3Hardware(commonpb.ProfileSource_PROFILE_SOURCE_REAL, "")
	hw.CpuPhysicalCores = 24
	resp, _ := srv.Register(context.Background(), &registrypb.RegisterRequest{WorkerId: "w", Address: "a", Hardware: hw})
	if resp.Accepted || !strings.Contains(resp.Message, "exceeds cpu_logical_cores") {
		t.Fatalf("expected rejection naming the problem: %+v", resp)
	}

	hw = schema3Hardware(commonpb.ProfileSource_PROFILE_SOURCE_UNSPECIFIED, "")
	resp, _ = srv.Register(context.Background(), &registrypb.RegisterRequest{WorkerId: "w", Address: "a", Hardware: hw})
	if resp.Accepted {
		t.Fatal("schema 3 without a profile source must be rejected")
	}
}

func TestHeartbeat_UnknownWorkerAsksForReregistration(t *testing.T) {
	srv := NewRegistryServer(registry.NewInMemoryRegistry())
	resp, err := srv.Heartbeat(context.Background(), &registrypb.HeartbeatRequest{WorkerId: "ghost"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Acknowledged || !resp.ReregisterRequired || resp.Message == "" {
		t.Fatalf("resp = %+v, want ack=false reregister_required=true", resp)
	}
}

func TestHeartbeat_InvalidMetricsRejectedWithoutReregister(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	srv := NewRegistryServer(store)
	if resp, _ := srv.Register(context.Background(), &registrypb.RegisterRequest{
		WorkerId: "w", Address: "a", Hardware: schema3Hardware(commonpb.ProfileSource_PROFILE_SOURCE_REAL, ""),
	}); !resp.Accepted {
		t.Fatal(resp.Message)
	}
	resp, _ := srv.Heartbeat(context.Background(), &registrypb.HeartbeatRequest{WorkerId: "w", CpuUtilizationPercent: pf64(180)})
	if resp.Acknowledged || resp.ReregisterRequired || !strings.Contains(resp.Message, "cpu_utilization_percent") {
		t.Fatalf("resp = %+v", resp)
	}
}

func TestHeartbeat_Schema3MetricsStored(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	srv := NewRegistryServer(store)
	if resp, _ := srv.Register(context.Background(), &registrypb.RegisterRequest{
		WorkerId: "w", Address: "a", Hardware: schema3Hardware(commonpb.ProfileSource_PROFILE_SOURCE_REAL, ""),
	}); !resp.Accepted {
		t.Fatal(resp.Message)
	}
	resp, _ := srv.Heartbeat(context.Background(), &registrypb.HeartbeatRequest{
		WorkerId:                 "w",
		CpuFrequencyCurrentMhz:   pf64(1700),
		CpuLoadAverage_1M:        pf64(0.75),
		SwapUsedBytes:            p64(2 << 30),
		SwapUtilizationPercent:   pf64(14),
		GpuMetrics:               []*commonpb.GPUMetrics{{GpuId: "luid:A", UtilizationPercent: pf64(5.2)}},
		MetricsCollectedAtUnixMs: 1793407907053,
	})
	if !resp.Acknowledged {
		t.Fatalf("heartbeat rejected: %s", resp.Message)
	}
	rec, _ := store.Get("w")
	if *rec.CPULoadAverage1m != 0.75 || *rec.SwapUsedBytes != 2<<30 || rec.MetricsCollectedAt.UnixMilli() != 1793407907053 {
		t.Errorf("metrics not stored: %+v", rec)
	}
	m := rec.GPUMetrics[0]
	if *m.UtilizationPercent != 5.2 || *m.SharedMemoryUsedBytes != 1<<30 {
		t.Errorf("gpu metric merge: %+v", m)
	}
}
