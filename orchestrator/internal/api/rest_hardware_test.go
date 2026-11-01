package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"orchestrator/internal/registry"
)

func ptrInt64(v int64) *int64       { return &v }
func ptrFloat64(v float64) *float64 { return &v }
func ptrBool(v bool) *bool          { return &v }

func TestHandleWorkers_IncludesPhase3HardwareFields(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	cudaKnownFalse := false
	if _, err := store.Register(registry.WorkerRecord{
		ID:                    "w1",
		Address:               "127.0.0.1:50100",
		Hostname:              "WORKER-PC",
		CPULogicalCores:       12,
		HardwareSchemaVersion: 2,
		CPUManufacturer:       "Intel",
		CPUModel:              "13th Gen Intel(R) Core(TM) i7-1355U",
		CPUPhysicalCores:      10,
		CPUArchitecture:       "AMD64",
		OSName:                "Windows",
		OSVersion:             "11",
		OSBuild:               "10.0.26200",
		GPUs: []registry.GPUInfo{
			{
				Vendor:           "Intel",
				Name:             "Intel(R) Iris(R) Xe Graphics",
				MemoryTotalBytes: ptrInt64(2147479552),
				MemoryType:       "shared",
				Backends:         []string{},
			},
		},
		TorchInstalled:        false,
		CUDAAvailable:         &cudaKnownFalse,
		CPUUtilizationPercent: ptrFloat64(12.5),
		RAMAvailableBytes:     ptrInt64(8 << 30),
	}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	srv := NewServer(store, "")

	req := httptest.NewRequest(http.MethodGet, "/api/workers", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}

	// Decode into a generic map first, to check field PRESENCE/ABSENCE
	// (the omitempty "never fabricate unknown values" contract), not
	// just values.
	var raw []map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &raw); err != nil {
		t.Fatalf("invalid JSON body: %v", err)
	}
	if len(raw) != 1 {
		t.Fatalf("got %d workers, want 1", len(raw))
	}
	worker := raw[0]

	if worker["cpu_physical_cores"] != float64(10) {
		t.Errorf("cpu_physical_cores = %v, want 10", worker["cpu_physical_cores"])
	}
	if worker["cpu_logical_cores"] != float64(12) {
		t.Errorf("cpu_logical_cores = %v, want 12", worker["cpu_logical_cores"])
	}
	if worker["cpu_physical_cores"] == worker["cpu_logical_cores"] {
		t.Error("physical and logical cores must not be reported as equal for this fixture")
	}

	gpus, ok := worker["gpus"].([]any)
	if !ok || len(gpus) != 1 {
		t.Fatalf("gpus = %v, want a one-element array", worker["gpus"])
	}
	gpu := gpus[0].(map[string]any)
	if gpu["memory_type"] != "shared" {
		t.Errorf("gpu memory_type = %v, want shared", gpu["memory_type"])
	}

	// cuda_available was explicitly set to a known false -- must be
	// PRESENT in the JSON (not omitted), since omitting it would make
	// "known false" indistinguishable from "unknown".
	if _, present := worker["cuda_available"]; !present {
		t.Error("cuda_available should be present when explicitly known (false), not omitted")
	}
	if worker["cuda_available"] != false {
		t.Errorf("cuda_available = %v, want false", worker["cuda_available"])
	}

	// directml_available was never set on this fixture -- must be
	// ABSENT, not a fabricated false.
	if _, present := worker["directml_available"]; present {
		t.Error("directml_available should be omitted when genuinely unknown, not reported as a fabricated value")
	}

	// cuda_device_count likewise never set -- must be absent.
	if _, present := worker["cuda_device_count"]; present {
		t.Error("cuda_device_count should be omitted when unknown")
	}
}

func TestHandleWorkers_GPUWithNilBackendsRendersAsEmptyArrayNotNull(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	if _, err := store.Register(registry.WorkerRecord{
		ID:      "w1",
		Address: "a",
		GPUs: []registry.GPUInfo{
			{Vendor: "Intel", Name: "Some iGPU", MemoryType: "shared", Backends: nil},
		},
	}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	srv := NewServer(store, "")

	req := httptest.NewRequest(http.MethodGet, "/api/workers", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	var raw []map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &raw); err != nil {
		t.Fatalf("invalid JSON body: %v", err)
	}
	gpus := raw[0]["gpus"].([]any)
	gpu := gpus[0].(map[string]any)
	backends, present := gpu["backends"]
	if !present {
		t.Fatal("backends key should be present")
	}
	if backends == nil {
		t.Error("backends should serialize as [] (empty array), not null, when Backends was a nil slice")
	}
	arr, ok := backends.([]any)
	if !ok || len(arr) != 0 {
		t.Errorf("backends = %v, want an empty array", backends)
	}
}

func TestHandleWorkers_CPUOnlyWorkerHasEmptyGPUArrayNotOmitted(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	if _, err := store.Register(registry.WorkerRecord{ID: "cpu-only", Address: "a"}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	srv := NewServer(store, "")

	req := httptest.NewRequest(http.MethodGet, "/api/workers", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	var raw []map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &raw); err != nil {
		t.Fatalf("invalid JSON body: %v", err)
	}
	gpus, ok := raw[0]["gpus"]
	if !ok {
		t.Fatal("gpus field should be present (as an empty array), not omitted, for a CPU-only worker")
	}
	arr, ok := gpus.([]any)
	if !ok || len(arr) != 0 {
		t.Errorf("gpus = %v, want an empty array", gpus)
	}
}
