// Package api exposes the orchestrator's REST surface, used by the
// dashboard (frontend/) and by operators/tests. Everything here reads
// from the real registry.Registry -- no hardcoded or fabricated data,
// per the project's frontend rule that all metrics must come from
// actual backend state.
//
// Phase 1 scope: GET /health and GET /api/workers. More endpoints
// (model partitioning, pipeline status, logs, inference submission)
// are added in later phases as the features they expose are built.
//
// Phase 3 adds (all read-only):
//
//	GET /api/workers?source=real|simulated|unknown   filter by profile source
//	GET /api/workers/{id}             one worker (same shape as the list)
//	GET /api/workers/{id}/hardware    static profile + classification
//	GET /api/workers/{id}/metrics     dynamic metrics only
//	GET /api/workers/{id}/resources   partition-planning view
//	GET /api/resources                partition-planning view, all workers
//
// See docs/HARDWARE_PROFILING.md for example responses.
package api

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"orchestrator/internal/capability"
	"orchestrator/internal/registry"
)

// Server serves the orchestrator's REST API.
type Server struct {
	store      registry.Registry
	startedAt  time.Time
	corsOrigin string
	capability capability.Config
}

// NewServer creates a REST API server backed by store. corsOrigin, if
// non-empty, is sent as the sole allowed Access-Control-Allow-Origin
// (the frontend dev server address); empty disables CORS headers.
// Capability classification uses capability.DefaultConfig().
func NewServer(store registry.Registry, corsOrigin string) *Server {
	return NewServerWithCapability(store, corsOrigin, capability.DefaultConfig())
}

// NewServerWithCapability is NewServer with explicit classification
// thresholds and safety margins (orchestrator.yaml "capability:").
func NewServerWithCapability(store registry.Registry, corsOrigin string, capCfg capability.Config) *Server {
	return &Server{store: store, startedAt: time.Now(), corsOrigin: corsOrigin, capability: capCfg}
}

// Handler returns the composed http.Handler for all routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/api/workers", s.handleWorkers)
	mux.HandleFunc("GET /api/workers/{id}", s.handleWorker)
	mux.HandleFunc("GET /api/workers/{id}/hardware", s.handleWorkerHardware)
	mux.HandleFunc("GET /api/workers/{id}/metrics", s.handleWorkerMetrics)
	mux.HandleFunc("GET /api/workers/{id}/resources", s.handleWorkerResources)
	mux.HandleFunc("GET /api/resources", s.handleResources)
	return s.withCORS(mux)
}

func (s *Server) withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.corsOrigin != "" {
			w.Header().Set("Access-Control-Allow-Origin", s.corsOrigin)
			w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		}
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// healthResponse is the real, live status of this orchestrator process.
type healthResponse struct {
	Status            string  `json:"status"`
	UptimeSeconds     float64 `json:"uptime_seconds"`
	RegisteredWorkers int     `json:"registered_workers"`
	OnlineWorkers     int     `json:"online_workers"`
	// Phase 3: online workers split by profile source, so a count of
	// "3 online" can never hide that two of them are simulated.
	OnlineRealWorkers      int `json:"online_real_workers"`
	OnlineSimulatedWorkers int `json:"online_simulated_workers"`
	OnlineUnknownWorkers   int `json:"online_unknown_source_workers"`
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	recs := s.store.List()
	resp := healthResponse{
		Status:            "ok",
		UptimeSeconds:     time.Since(s.startedAt).Seconds(),
		RegisteredWorkers: len(recs),
	}
	for _, rec := range recs {
		if !rec.Online {
			continue
		}
		resp.OnlineWorkers++
		switch rec.ProfileSource {
		case registry.ProfileSourceReal:
			resp.OnlineRealWorkers++
		case registry.ProfileSourceSimulated:
			resp.OnlineSimulatedWorkers++
		default:
			resp.OnlineUnknownWorkers++
		}
	}
	writeJSON(w, http.StatusOK, resp)
}

// gpuView is the JSON shape of one GPU in a worker's inventory.
// MemoryTotalBytes is a pointer so JSON omits it (rather than showing
// a fabricated 0) when the registry genuinely doesn't know it.
type gpuView struct {
	Vendor           string   `json:"vendor"`
	Name             string   `json:"name"`
	MemoryTotalBytes *int64   `json:"memory_total_bytes,omitempty"`
	MemoryType       string   `json:"memory_type"`
	Backends         []string `json:"backends"`
	DriverVersion    string   `json:"driver_version"`

	// --- schema 3 ---
	ID                   string `json:"id,omitempty"`
	DedicatedMemoryBytes *int64 `json:"dedicated_memory_bytes,omitempty"`
	SharedMemoryBytes    *int64 `json:"shared_memory_bytes,omitempty"`
	MemorySource         string `json:"memory_source,omitempty"`
	Integrated           bool   `json:"integrated"`
	// Live, from the most recent GPU metrics sample (omitted if none).
	UtilizationPercent       *float64 `json:"utilization_percent,omitempty"`
	DedicatedMemoryUsedBytes *int64   `json:"dedicated_memory_used_bytes,omitempty"`
	SharedMemoryUsedBytes    *int64   `json:"shared_memory_used_bytes,omitempty"`
}

// backendView is one probed inference backend.
type backendView struct {
	Name        string   `json:"name"`
	Available   *bool    `json:"available,omitempty"`
	Usable      bool     `json:"usable"`
	DeviceCount int32    `json:"device_count"`
	Version     string   `json:"version,omitempty"`
	Devices     []string `json:"devices"`
	Tested      bool     `json:"tested"`
	TestPassed  *bool    `json:"test_passed,omitempty"`
	TestGFLOPS  *float64 `json:"test_gflops,omitempty"`
	Detail      string   `json:"detail,omitempty"`
}

// workerView is the JSON shape of one worker, as actually held in the
// registry -- never fabricated. Online and LastHeartbeatAgeSeconds are
// computed fresh on every request (never cached), so a worker that has
// gone stale shows up as such immediately, without waiting on any
// background process.
//
// Phase 3 fields below use pointers for anything that can be
// genuinely unknown (CUDAAvailable, CPUUtilizationPercent, etc.), so
// JSON omits them rather than showing a fabricated false/0 -- a
// client (the dashboard, a test) can distinguish "known false/zero"
// from "not reported" by checking for the key's presence.
type workerView struct {
	ID      string `json:"id"`
	Address string `json:"address"`
	// ObservedPeerIP (Phase 8): where the worker's Register call
	// actually came from, as seen by this orchestrator. "" if unknown.
	ObservedPeerIP          string    `json:"observed_peer_ip"`
	Hostname                string    `json:"hostname"`
	CPULogicalCores         int32     `json:"cpu_logical_cores"`
	TotalRAMBytes           int64     `json:"total_ram_bytes"`
	HasCUDAGPU              bool      `json:"has_cuda_gpu"`
	GPUName                 string    `json:"gpu_name"`
	GPUVRAMBytes            int64     `json:"gpu_vram_bytes"`
	OS                      string    `json:"os"`
	PythonVersion           string    `json:"python_version"`
	Status                  string    `json:"status"`
	Online                  bool      `json:"online"`
	LastHeartbeatAgeSeconds float64   `json:"last_heartbeat_age_seconds"`
	RegisteredAt            time.Time `json:"registered_at"`
	LastHeartbeat           time.Time `json:"last_heartbeat"`

	// --- Phase 3 ---
	HardwareSchemaVersion int32     `json:"hardware_schema_version"`
	CPUManufacturer       string    `json:"cpu_manufacturer"`
	CPUModel              string    `json:"cpu_model"`
	CPUPhysicalCores      int32     `json:"cpu_physical_cores"`
	CPUArchitecture       string    `json:"cpu_architecture"`
	OSName                string    `json:"os_name"`
	OSVersion             string    `json:"os_version"`
	OSBuild               string    `json:"os_build"`
	GPUs                  []gpuView `json:"gpus"`
	TorchInstalled        bool      `json:"torch_installed"`
	TorchVersion          string    `json:"torch_version,omitempty"`
	CPUBackendAvailable   bool      `json:"cpu_backend_available"`
	CUDAAvailable         *bool     `json:"cuda_available,omitempty"`
	CUDADeviceCount       *int32    `json:"cuda_device_count,omitempty"`
	DirectMLAvailable     *bool     `json:"directml_available,omitempty"`
	CPUUtilizationPercent *float64  `json:"cpu_utilization_percent,omitempty"`
	RAMAvailableBytes     *int64    `json:"ram_available_bytes,omitempty"`
	RAMUsedBytes          *int64    `json:"ram_used_bytes,omitempty"`
	RAMUtilizationPercent *float64  `json:"ram_utilization_percent,omitempty"`

	// --- schema 3 ---
	ProfileSource string `json:"profile_source"`
	// Present for anything not measured as REAL hardware.
	ProfileNotice        string     `json:"profile_notice,omitempty"`
	SimulatedProfileName string     `json:"simulated_profile_name,omitempty"`
	ProfileCollectedAt   *time.Time `json:"profile_collected_at,omitempty"`
	HardwareUpdatedAt    time.Time  `json:"hardware_updated_at"`
	MetricsUpdatedAt     time.Time  `json:"metrics_updated_at"`
	MetricsCollectedAt   *time.Time `json:"metrics_collected_at,omitempty"`

	CPUFrequencyCurrentMHz *float64 `json:"cpu_frequency_current_mhz,omitempty"`
	CPUFrequencyMaxMHz     *float64 `json:"cpu_frequency_max_mhz,omitempty"`
	CPULoadAverage1m       *float64 `json:"cpu_load_average_1m,omitempty"`
	SwapTotalBytes         *int64   `json:"swap_total_bytes,omitempty"`
	SwapUsedBytes          *int64   `json:"swap_used_bytes,omitempty"`
	SwapUtilizationPercent *float64 `json:"swap_utilization_percent,omitempty"`

	GPUCount              int                       `json:"gpu_count"`
	BackendProbePending   bool                      `json:"backend_probe_pending"`
	BackendCapabilities   []backendView             `json:"backend_capabilities"`
	TorchCUDABuildVersion string                    `json:"torch_cuda_build_version,omitempty"`
	Classification        capability.Classification `json:"classification"`
}

func profileNotice(src registry.ProfileSource) string {
	switch src {
	case registry.ProfileSourceReal:
		return ""
	case registry.ProfileSourceSimulated:
		return registry.SimulatedNotice
	default:
		return "profile source not reported (worker predates hardware schema 3); not verified as real hardware"
	}
}

func optTime(t time.Time) *time.Time {
	if t.IsZero() {
		return nil
	}
	return &t
}

func (s *Server) handleWorkers(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	filter := strings.ToUpper(strings.TrimSpace(r.URL.Query().Get("source")))
	switch filter {
	case "", string(registry.ProfileSourceReal), string(registry.ProfileSourceSimulated), string(registry.ProfileSourceUnknown):
	default:
		http.Error(w, "source must be one of real, simulated, unknown", http.StatusBadRequest)
		return
	}
	recs := s.store.List()
	out := make([]workerView, 0, len(recs))
	for _, rec := range recs {
		if filter != "" && string(rec.ProfileSource) != filter {
			continue
		}
		out = append(out, s.toWorkerView(rec))
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) toWorkerView(rec registry.WorkerRecord) workerView {
	return workerView{
		ID:                      rec.ID,
		Address:                 rec.Address,
		ObservedPeerIP:          rec.ObservedPeerIP,
		Hostname:                rec.Hostname,
		CPULogicalCores:         rec.CPULogicalCores,
		TotalRAMBytes:           rec.TotalRAMBytes,
		HasCUDAGPU:              rec.HasCUDAGPU,
		GPUName:                 rec.GPUName,
		GPUVRAMBytes:            rec.GPUVRAMBytes,
		OS:                      rec.OS,
		PythonVersion:           rec.PythonVersion,
		Status:                  rec.Status,
		Online:                  rec.Online,
		LastHeartbeatAgeSeconds: time.Since(rec.LastHeartbeat).Seconds(),
		RegisteredAt:            rec.RegisteredAt,
		LastHeartbeat:           rec.LastHeartbeat,

		HardwareSchemaVersion: rec.HardwareSchemaVersion,
		CPUManufacturer:       rec.CPUManufacturer,
		CPUModel:              rec.CPUModel,
		CPUPhysicalCores:      rec.CPUPhysicalCores,
		CPUArchitecture:       rec.CPUArchitecture,
		OSName:                rec.OSName,
		OSVersion:             rec.OSVersion,
		OSBuild:               rec.OSBuild,
		GPUs:                  toGPUViews(rec.GPUs, rec.GPUMetrics),
		TorchInstalled:        rec.TorchInstalled,
		TorchVersion:          rec.TorchVersion,
		CPUBackendAvailable:   rec.CPUBackendAvailable,
		CUDAAvailable:         rec.CUDAAvailable,
		CUDADeviceCount:       rec.CUDADeviceCount,
		DirectMLAvailable:     rec.DirectMLAvailable,
		CPUUtilizationPercent: rec.CPUUtilizationPercent,
		RAMAvailableBytes:     rec.RAMAvailableBytes,
		RAMUsedBytes:          rec.RAMUsedBytes,
		RAMUtilizationPercent: rec.RAMUtilizationPercent,

		ProfileSource:          string(rec.ProfileSource),
		ProfileNotice:          profileNotice(rec.ProfileSource),
		SimulatedProfileName:   rec.SimulatedProfileName,
		ProfileCollectedAt:     optTime(rec.ProfileCollectedAt),
		HardwareUpdatedAt:      rec.HardwareUpdatedAt,
		MetricsUpdatedAt:       rec.MetricsUpdatedAt,
		MetricsCollectedAt:     optTime(rec.MetricsCollectedAt),
		CPUFrequencyCurrentMHz: rec.CPUFrequencyCurrentMHz,
		CPUFrequencyMaxMHz:     rec.CPUFrequencyMaxMHz,
		CPULoadAverage1m:       rec.CPULoadAverage1m,
		SwapTotalBytes:         rec.SwapTotalBytes,
		SwapUsedBytes:          rec.SwapUsedBytes,
		SwapUtilizationPercent: rec.SwapUtilizationPercent,
		GPUCount:               len(rec.GPUs),
		BackendProbePending:    rec.BackendProbePending,
		BackendCapabilities:    toBackendViews(rec.BackendCapabilities),
		TorchCUDABuildVersion:  rec.TorchCUDABuildVersion,
		Classification:         capability.Classify(rec, s.capability),
	}
}

func toBackendViews(caps []registry.BackendCapability) []backendView {
	out := make([]backendView, 0, len(caps))
	for _, b := range caps {
		devices := b.Devices
		if devices == nil {
			devices = []string{}
		}
		out = append(out, backendView{
			Name:        b.Name,
			Available:   b.Available,
			Usable:      b.Usable(),
			DeviceCount: b.DeviceCount,
			Version:     b.Version,
			Devices:     devices,
			Tested:      b.Tested,
			TestPassed:  b.TestPassed,
			TestGFLOPS:  b.TestGFLOPS,
			Detail:      b.Detail,
		})
	}
	return out
}

func toGPUViews(gpus []registry.GPUInfo, metrics []registry.GPUMetrics) []gpuView {
	byID := make(map[string]registry.GPUMetrics, len(metrics))
	for _, m := range metrics {
		byID[m.GPUID] = m
	}
	out := make([]gpuView, 0, len(gpus))
	for _, g := range gpus {
		backends := g.Backends
		if backends == nil {
			// A nil slice marshals as JSON null; an empty-but-non-nil
			// one marshals as []. "No backends detected" should look
			// like an empty list, not null, for consistency with how
			// the top-level gpus array itself is presented.
			backends = []string{}
		}
		m := byID[g.ID]
		if g.ID == "" {
			m = registry.GPUMetrics{}
		}
		out = append(out, gpuView{
			Vendor:           g.Vendor,
			Name:             g.Name,
			MemoryTotalBytes: g.MemoryTotalBytes,
			MemoryType:       g.MemoryType,
			Backends:         backends,
			DriverVersion:    g.DriverVersion,

			ID:                       g.ID,
			DedicatedMemoryBytes:     g.DedicatedMemoryBytes,
			SharedMemoryBytes:        g.SharedMemoryBytes,
			MemorySource:             g.MemorySource,
			Integrated:               g.Integrated,
			UtilizationPercent:       m.UtilizationPercent,
			DedicatedMemoryUsedBytes: m.DedicatedMemoryUsedBytes,
			SharedMemoryUsedBytes:    m.SharedMemoryUsedBytes,
		})
	}
	return out
}

// lookup returns the worker named by the {id} path value, writing a
// 404 itself if it's unknown.
func (s *Server) lookup(w http.ResponseWriter, r *http.Request) (registry.WorkerRecord, bool) {
	rec, ok := s.store.Get(r.PathValue("id"))
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "worker not found", "worker_id": r.PathValue("id")})
	}
	return rec, ok
}

func (s *Server) handleWorker(w http.ResponseWriter, r *http.Request) {
	if rec, ok := s.lookup(w, r); ok {
		writeJSON(w, http.StatusOK, s.toWorkerView(rec))
	}
}

// hardwareView is the static part of a worker's profile -- what
// changes only when the worker re-collects its full snapshot.
type hardwareView struct {
	WorkerID              string                    `json:"worker_id"`
	ProfileSource         string                    `json:"profile_source"`
	ProfileNotice         string                    `json:"profile_notice,omitempty"`
	SimulatedProfileName  string                    `json:"simulated_profile_name,omitempty"`
	HardwareSchemaVersion int32                     `json:"hardware_schema_version"`
	ProfileCollectedAt    *time.Time                `json:"profile_collected_at,omitempty"`
	HardwareUpdatedAt     time.Time                 `json:"hardware_updated_at"`
	Hostname              string                    `json:"hostname"`
	OS                    map[string]string         `json:"os"`
	CPU                   cpuStaticView             `json:"cpu"`
	RAMTotalBytes         int64                     `json:"ram_total_bytes"`
	GPUs                  []gpuView                 `json:"gpus"`
	TorchInstalled        bool                      `json:"torch_installed"`
	TorchVersion          string                    `json:"torch_version,omitempty"`
	TorchCUDABuildVersion string                    `json:"torch_cuda_build_version,omitempty"`
	BackendProbePending   bool                      `json:"backend_probe_pending"`
	BackendCapabilities   []backendView             `json:"backend_capabilities"`
	Classification        capability.Classification `json:"classification"`
}

type cpuStaticView struct {
	Manufacturer    string   `json:"manufacturer"`
	Model           string   `json:"model"`
	Architecture    string   `json:"architecture"`
	PhysicalCores   int32    `json:"physical_cores"`
	LogicalCores    int32    `json:"logical_cores"`
	MaxFrequencyMHz *float64 `json:"max_frequency_mhz,omitempty"`
}

func (s *Server) handleWorkerHardware(w http.ResponseWriter, r *http.Request) {
	rec, ok := s.lookup(w, r)
	if !ok {
		return
	}
	gpus := toGPUViews(rec.GPUs, nil) // static only: no live metrics here
	writeJSON(w, http.StatusOK, hardwareView{
		WorkerID:              rec.ID,
		ProfileSource:         string(rec.ProfileSource),
		ProfileNotice:         profileNotice(rec.ProfileSource),
		SimulatedProfileName:  rec.SimulatedProfileName,
		HardwareSchemaVersion: rec.HardwareSchemaVersion,
		ProfileCollectedAt:    optTime(rec.ProfileCollectedAt),
		HardwareUpdatedAt:     rec.HardwareUpdatedAt,
		Hostname:              rec.Hostname,
		OS:                    map[string]string{"name": rec.OSName, "version": rec.OSVersion, "build": rec.OSBuild, "summary": rec.OS},
		CPU: cpuStaticView{
			Manufacturer:    rec.CPUManufacturer,
			Model:           rec.CPUModel,
			Architecture:    rec.CPUArchitecture,
			PhysicalCores:   rec.CPUPhysicalCores,
			LogicalCores:    rec.CPULogicalCores,
			MaxFrequencyMHz: rec.CPUFrequencyMaxMHz,
		},
		RAMTotalBytes:         rec.TotalRAMBytes,
		GPUs:                  gpus,
		TorchInstalled:        rec.TorchInstalled,
		TorchVersion:          rec.TorchVersion,
		TorchCUDABuildVersion: rec.TorchCUDABuildVersion,
		BackendProbePending:   rec.BackendProbePending,
		BackendCapabilities:   toBackendViews(rec.BackendCapabilities),
		Classification:        capability.Classify(rec, s.capability),
	})
}

// metricsView is the dynamic part of a worker's profile.
type metricsView struct {
	WorkerID           string           `json:"worker_id"`
	ProfileSource      string           `json:"profile_source"`
	ProfileNotice      string           `json:"profile_notice,omitempty"`
	Online             bool             `json:"online"`
	Status             string           `json:"status"`
	MetricsUpdatedAt   time.Time        `json:"metrics_updated_at"`
	MetricsAgeSeconds  float64          `json:"metrics_age_seconds"`
	MetricsCollectedAt *time.Time       `json:"metrics_collected_at,omitempty"`
	CPU                map[string]any   `json:"cpu"`
	RAM                map[string]any   `json:"ram"`
	Swap               map[string]any   `json:"swap"`
	GPUs               []map[string]any `json:"gpus"`
}

// putIf adds v to m only when it's a non-nil pointer, so unknown
// metrics are absent from the JSON rather than shown as 0.
func putIf[T any](m map[string]any, key string, v *T) {
	if v != nil {
		m[key] = *v
	}
}

func (s *Server) handleWorkerMetrics(w http.ResponseWriter, r *http.Request) {
	rec, ok := s.lookup(w, r)
	if !ok {
		return
	}
	cpu := map[string]any{}
	putIf(cpu, "utilization_percent", rec.CPUUtilizationPercent)
	putIf(cpu, "frequency_current_mhz", rec.CPUFrequencyCurrentMHz)
	putIf(cpu, "load_average_1m", rec.CPULoadAverage1m)
	ram := map[string]any{"total_bytes": rec.TotalRAMBytes}
	putIf(ram, "available_bytes", rec.RAMAvailableBytes)
	putIf(ram, "used_bytes", rec.RAMUsedBytes)
	putIf(ram, "utilization_percent", rec.RAMUtilizationPercent)
	swap := map[string]any{}
	putIf(swap, "total_bytes", rec.SwapTotalBytes)
	putIf(swap, "used_bytes", rec.SwapUsedBytes)
	putIf(swap, "utilization_percent", rec.SwapUtilizationPercent)
	gpus := make([]map[string]any, 0, len(rec.GPUMetrics))
	for _, m := range rec.GPUMetrics {
		g := map[string]any{"gpu_id": m.GPUID}
		putIf(g, "utilization_percent", m.UtilizationPercent)
		putIf(g, "dedicated_memory_used_bytes", m.DedicatedMemoryUsedBytes)
		putIf(g, "shared_memory_used_bytes", m.SharedMemoryUsedBytes)
		gpus = append(gpus, g)
	}
	writeJSON(w, http.StatusOK, metricsView{
		WorkerID:           rec.ID,
		ProfileSource:      string(rec.ProfileSource),
		ProfileNotice:      profileNotice(rec.ProfileSource),
		Online:             rec.Online,
		Status:             rec.Status,
		MetricsUpdatedAt:   rec.MetricsUpdatedAt,
		MetricsAgeSeconds:  time.Since(rec.MetricsUpdatedAt).Seconds(),
		MetricsCollectedAt: optTime(rec.MetricsCollectedAt),
		CPU:                cpu,
		RAM:                ram,
		Swap:               swap,
		GPUs:               gpus,
	})
}

func (s *Server) handleWorkerResources(w http.ResponseWriter, r *http.Request) {
	if rec, ok := s.lookup(w, r); ok {
		writeJSON(w, http.StatusOK, capability.ComputeResources(rec, s.capability, time.Now()))
	}
}

// resourcesResponse keeps real and simulated workers in separate lists
// so nothing consuming it can accidentally total them together.
type resourcesResponse struct {
	Real      []capability.Resources `json:"real"`
	Simulated []capability.Resources `json:"simulated"`
	Unknown   []capability.Resources `json:"unknown_source"`
	Notice    string                 `json:"notice"`
}

func (s *Server) handleResources(w http.ResponseWriter, r *http.Request) {
	resp := resourcesResponse{
		Real:      []capability.Resources{},
		Simulated: []capability.Resources{},
		Unknown:   []capability.Resources{},
		Notice:    "entries under \"simulated\" are " + registry.SimulatedNotice + "; do not total them with \"real\"",
	}
	now := time.Now()
	for _, rec := range s.store.List() {
		res := capability.ComputeResources(rec, s.capability, now)
		switch rec.ProfileSource {
		case registry.ProfileSourceReal:
			resp.Real = append(resp.Real, res)
		case registry.ProfileSourceSimulated:
			resp.Simulated = append(resp.Simulated, res)
		default:
			resp.Unknown = append(resp.Unknown, res)
		}
	}
	writeJSON(w, http.StatusOK, resp)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
