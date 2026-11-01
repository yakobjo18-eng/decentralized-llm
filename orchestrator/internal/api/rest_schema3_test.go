package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"orchestrator/internal/registry"
)

func realWorker(id string) registry.WorkerRecord {
	return registry.WorkerRecord{
		ID: id, Address: "127.0.0.1:1", Hostname: "YakobsPC", HardwareSchemaVersion: 3,
		ProfileSource: registry.ProfileSourceReal, ProfileCollectedAt: time.UnixMilli(1793407905533),
		CPULogicalCores: 12, CPUPhysicalCores: 10, CPUModel: "13th Gen Intel(R) Core(TM) i7-1355U",
		TotalRAMBytes: 16876888064, RAMAvailableBytes: ptrInt64(3338182656), CPUBackendAvailable: true, TorchInstalled: true,
		CPUFrequencyMaxMHz: ptrFloat64(1700),
		GPUs: []registry.GPUInfo{{
			ID: "luid:A", Vendor: "Intel", Name: "Intel(R) Iris(R) Xe Graphics", MemoryType: "shared", Integrated: true,
			DedicatedMemoryBytes: ptrInt64(134217728), SharedMemoryBytes: ptrInt64(8438444032), MemorySource: "dxgi",
		}},
		GPUMetrics: []registry.GPUMetrics{{GPUID: "luid:A", UtilizationPercent: ptrFloat64(5.2), SharedMemoryUsedBytes: ptrInt64(1 << 30)}},
		BackendCapabilities: []registry.BackendCapability{
			{Name: "cpu", Available: ptrBool(true), Tested: true, TestPassed: ptrBool(true), TestGFLOPS: ptrFloat64(155.89)},
			{Name: "cuda", Available: ptrBool(false), Version: "12.6"},
		},
	}
}

func simulatedWorker(id string) registry.WorkerRecord {
	return registry.WorkerRecord{
		ID: id, Address: "127.0.0.1:2", Hostname: "sim-profile-b", HardwareSchemaVersion: 3,
		ProfileSource: registry.ProfileSourceSimulated, SimulatedProfileName: "profile-b-gpu-worker",
		CPULogicalCores: 12, CPUPhysicalCores: 6, TotalRAMBytes: 32 << 30, RAMAvailableBytes: ptrInt64(24 << 30),
		TorchInstalled: true, CPUBackendAvailable: true,
		GPUs: []registry.GPUInfo{{ID: "sim:b:0", Vendor: "NVIDIA", Name: "Simulated GPU", MemoryType: "dedicated",
			MemoryTotalBytes: ptrInt64(8 << 30), DedicatedMemoryBytes: ptrInt64(8 << 30), Backends: []string{"CUDA"}}},
		GPUMetrics: []registry.GPUMetrics{{GPUID: "sim:b:0", DedicatedMemoryUsedBytes: ptrInt64(1 << 30)}},
		BackendCapabilities: []registry.BackendCapability{
			{Name: "cpu", Available: ptrBool(true)},
			{Name: "cuda", Available: ptrBool(true), DeviceCount: 1},
		},
	}
}

func setup(t *testing.T, recs ...registry.WorkerRecord) http.Handler {
	t.Helper()
	store := registry.NewInMemoryRegistry()
	for _, r := range recs {
		if _, err := store.Register(r); err != nil {
			t.Fatalf("Register(%s): %v", r.ID, err)
		}
	}
	return NewServer(store, "").Handler()
}

func get(t *testing.T, h http.Handler, path string) (int, map[string]any, []any) {
	t.Helper()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
	var v any
	if err := json.Unmarshal(rec.Body.Bytes(), &v); err != nil && rec.Code == http.StatusOK {
		t.Fatalf("%s: bad JSON %q", path, rec.Body.String())
	}
	obj, _ := v.(map[string]any)
	arr, _ := v.([]any)
	return rec.Code, obj, arr
}

func TestWorkers_Schema3FieldsAndSourceLabels(t *testing.T) {
	h := setup(t, realWorker("real-1"), simulatedWorker("sim-1"), registry.WorkerRecord{ID: "old-1", Address: "x"})
	_, _, arr := get(t, h, "/api/workers")
	by := map[string]map[string]any{}
	for _, w := range arr {
		m := w.(map[string]any)
		by[m["id"].(string)] = m
	}

	real := by["real-1"]
	if real["profile_source"] != "REAL" {
		t.Errorf("real profile_source = %v", real["profile_source"])
	}
	if _, has := real["profile_notice"]; has {
		t.Error("a REAL worker must not carry a notice")
	}
	if real["profile_collected_at"] == nil || real["hardware_updated_at"] == nil || real["metrics_updated_at"] == nil {
		t.Error("timestamps missing")
	}
	classes := real["classification"].(map[string]any)["classes"].([]any)
	if len(classes) != 1 || classes[0] != "CPU_ONLY" {
		t.Errorf("real classes = %v", classes)
	}
	gpu := real["gpus"].([]any)[0].(map[string]any)
	if _, has := gpu["memory_total_bytes"]; has {
		t.Error("integrated GPU must not show memory_total_bytes")
	}
	if gpu["integrated"] != true || gpu["utilization_percent"] != 5.2 || gpu["shared_memory_bytes"] != float64(8438444032) {
		t.Errorf("gpu view = %v", gpu)
	}
	if real["gpu_count"] != float64(1) {
		t.Errorf("gpu_count = %v", real["gpu_count"])
	}

	sim := by["sim-1"]
	if sim["profile_source"] != "SIMULATED" || sim["profile_notice"] != registry.SimulatedNotice || sim["simulated_profile_name"] != "profile-b-gpu-worker" {
		t.Errorf("simulated labels wrong: %v %v %v", sim["profile_source"], sim["profile_notice"], sim["simulated_profile_name"])
	}

	old := by["old-1"]
	if old["profile_source"] != "UNKNOWN" || old["profile_notice"] == nil {
		t.Errorf("pre-schema-3 worker must be UNKNOWN with a notice, got %v / %v", old["profile_source"], old["profile_notice"])
	}
}

func TestWorkers_SourceFilter(t *testing.T) {
	h := setup(t, realWorker("real-1"), simulatedWorker("sim-1"))
	for query, want := range map[string]string{"real": "real-1", "SIMULATED": "sim-1"} {
		_, _, arr := get(t, h, "/api/workers?source="+query)
		if len(arr) != 1 || arr[0].(map[string]any)["id"] != want {
			t.Errorf("source=%s -> %v", query, arr)
		}
	}
	if code, _, _ := get(t, h, "/api/workers?source=imaginary"); code != http.StatusBadRequest {
		t.Errorf("bad filter status = %d", code)
	}
}

func TestWorkerEndpoints(t *testing.T) {
	h := setup(t, realWorker("real-1"))

	if code, obj, _ := get(t, h, "/api/workers/missing"); code != http.StatusNotFound || obj["error"] == nil {
		t.Errorf("missing worker: %d %v", code, obj)
	}
	for _, suffix := range []string{"/hardware", "/metrics", "/resources"} {
		if code, _, _ := get(t, h, "/api/workers/missing"+suffix); code != http.StatusNotFound {
			t.Errorf("missing%s: %d", suffix, code)
		}
	}

	code, w, _ := get(t, h, "/api/workers/real-1")
	if code != http.StatusOK || w["id"] != "real-1" {
		t.Fatalf("single worker: %d %v", code, w)
	}

	_, hw, _ := get(t, h, "/api/workers/real-1/hardware")
	cpu := hw["cpu"].(map[string]any)
	if cpu["physical_cores"] != float64(10) || cpu["logical_cores"] != float64(12) || hw["hardware_schema_version"] != float64(3) {
		t.Errorf("hardware view cpu = %v", cpu)
	}
	if _, has := hw["gpus"].([]any)[0].(map[string]any)["utilization_percent"]; has {
		t.Error("/hardware is static only: no live GPU metrics")
	}
	if len(hw["backend_capabilities"].([]any)) != 2 {
		t.Error("backend capabilities missing from /hardware")
	}

	_, m, _ := get(t, h, "/api/workers/real-1/metrics")
	if m["ram"].(map[string]any)["available_bytes"] != float64(3338182656) {
		t.Errorf("metrics ram = %v", m["ram"])
	}
	if _, has := m["cpu"].(map[string]any)["utilization_percent"]; has {
		t.Error("unreported cpu utilization must be absent, not 0")
	}
	if g := m["gpus"].([]any)[0].(map[string]any); g["utilization_percent"] != 5.2 {
		t.Errorf("metrics gpu = %v", g)
	}

	_, res, _ := get(t, h, "/api/workers/real-1/resources")
	if res["ram_usable_bytes"] == nil || res["ram_usable_bytes"].(float64) >= res["ram_available_bytes"].(float64) {
		t.Errorf("usable RAM must be below available: %v", res)
	}
	if res["cpu_measured_gflops"] != 155.89 {
		t.Errorf("gflops = %v", res["cpu_measured_gflops"])
	}
	if _, has := res["total_usable_vram_bytes"]; has {
		t.Error("Computer 1 has no usable VRAM")
	}
}

func TestResources_SeparatesRealAndSimulated(t *testing.T) {
	h := setup(t, realWorker("real-1"), simulatedWorker("sim-1"))
	_, obj, _ := get(t, h, "/api/resources")
	real, sim := obj["real"].([]any), obj["simulated"].([]any)
	if len(real) != 1 || len(sim) != 1 || len(obj["unknown_source"].([]any)) != 0 {
		t.Fatalf("resources = %v", obj)
	}
	if sim[0].(map[string]any)["total_usable_vram_bytes"] == nil {
		t.Error("simulated GPU worker should expose usable VRAM")
	}
	if obj["notice"] == "" {
		t.Error("notice missing")
	}
}

func TestHealth_CountsBySource(t *testing.T) {
	h := setup(t, realWorker("real-1"), simulatedWorker("sim-1"), simulatedWorker("sim-2"), registry.WorkerRecord{ID: "old", Address: "x"})
	_, obj, _ := get(t, h, "/health")
	if obj["online_workers"] != float64(4) || obj["online_real_workers"] != float64(1) ||
		obj["online_simulated_workers"] != float64(2) || obj["online_unknown_source_workers"] != float64(1) {
		t.Errorf("health = %v", obj)
	}
}

func TestNewEndpointsAreReadOnly(t *testing.T) {
	h := setup(t, realWorker("real-1"))
	for _, path := range []string{"/api/workers/real-1", "/api/workers/real-1/hardware", "/api/resources"} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, path, nil))
		if rec.Code != http.StatusMethodNotAllowed {
			t.Errorf("POST %s = %d, want 405", path, rec.Code)
		}
	}
}
