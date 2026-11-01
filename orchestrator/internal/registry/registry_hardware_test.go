package registry

import (
	"testing"
	"time"
)

func f64(v float64) *float64 { return &v }
func i64(v int64) *int64     { return &v }
func i32(v int32) *int32     { return &v }
func b(v bool) *bool         { return &v }

func TestRegister_StoresPhase3HardwareFields(t *testing.T) {
	r := NewInMemoryRegistry()

	_, err := r.Register(WorkerRecord{
		ID:                    "w1",
		Address:               "127.0.0.1:50100",
		HardwareSchemaVersion: 2,
		CPUManufacturer:       "Intel",
		CPUModel:              "13th Gen Intel(R) Core(TM) i7-1355U",
		CPUPhysicalCores:      10,
		CPULogicalCores:       12,
		CPUArchitecture:       "AMD64",
		OSName:                "Windows",
		OSVersion:             "11",
		OSBuild:               "10.0.26200",
		GPUs: []GPUInfo{
			{
				Vendor:           "Intel",
				Name:             "Intel(R) Iris(R) Xe Graphics",
				MemoryTotalBytes: i64(2147479552),
				MemoryType:       "shared",
				Backends:         []string{},
				DriverVersion:    "32.0.101.6881",
			},
		},
		TorchInstalled:        false,
		CPUBackendAvailable:   false,
		CUDAAvailable:         nil, // unknown -- torch not installed
		CPUUtilizationPercent: f64(12.5),
		RAMAvailableBytes:     i64(8 << 30),
		RAMUsedBytes:          i64(7 << 30),
		RAMUtilizationPercent: f64(46.6),
	})
	if err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	rec, ok := r.Get("w1")
	if !ok {
		t.Fatal("worker not found after Register()")
	}

	if rec.CPUPhysicalCores != 10 || rec.CPULogicalCores != 12 {
		t.Errorf("physical=%d logical=%d, want 10/12 (must not conflate them)", rec.CPUPhysicalCores, rec.CPULogicalCores)
	}
	if rec.CPUPhysicalCores == rec.CPULogicalCores {
		t.Error("physical and logical core counts must not be equal here -- that would mean they were conflated")
	}
	if len(rec.GPUs) != 1 || rec.GPUs[0].MemoryType != "shared" {
		t.Errorf("GPUs = %+v, want one shared-memory GPU", rec.GPUs)
	}
	if rec.CUDAAvailable != nil {
		t.Errorf("CUDAAvailable = %v, want nil (unknown) when torch isn't installed", *rec.CUDAAvailable)
	}
	if rec.CPUUtilizationPercent == nil || *rec.CPUUtilizationPercent != 12.5 {
		t.Errorf("CPUUtilizationPercent = %v, want 12.5", rec.CPUUtilizationPercent)
	}
}

func TestHeartbeatWithMetrics_UpdatesOnlyProvidedFields(t *testing.T) {
	r := NewInMemoryRegistry()
	if _, err := r.Register(WorkerRecord{
		ID:                    "w1",
		Address:               "a",
		CPUUtilizationPercent: f64(10.0),
		RAMUsedBytes:          i64(1000),
	}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	// Heartbeat reports ONLY cpu_utilization_percent this beat (e.g.
	// RAM sampling failed on the worker that beat) -- RAMUsedBytes
	// must be left untouched, not zeroed.
	err := r.HeartbeatWithMetrics("w1", "idle", time.Now(), LiveMetrics{
		CPUUtilizationPercent: f64(55.0),
	})
	if err != nil {
		t.Fatalf("HeartbeatWithMetrics() error = %v", err)
	}

	rec, _ := r.Get("w1")
	if rec.CPUUtilizationPercent == nil || *rec.CPUUtilizationPercent != 55.0 {
		t.Errorf("CPUUtilizationPercent = %v, want 55.0 (updated)", rec.CPUUtilizationPercent)
	}
	if rec.RAMUsedBytes == nil || *rec.RAMUsedBytes != 1000 {
		t.Errorf("RAMUsedBytes = %v, want 1000 (untouched, not zeroed)", rec.RAMUsedBytes)
	}
}

func TestHeartbeat_LegacyMethodDoesNotTouchLiveMetrics(t *testing.T) {
	r := NewInMemoryRegistry()
	if _, err := r.Register(WorkerRecord{
		ID:                    "w1",
		Address:               "a",
		CPUUtilizationPercent: f64(42.0),
	}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	// The plain Heartbeat (Phase 1/2 signature) must behave exactly as
	// it always did: it must NOT wipe out live metrics that a
	// HeartbeatWithMetrics call (or Register) previously set.
	if err := r.Heartbeat("w1", "busy", time.Now()); err != nil {
		t.Fatalf("Heartbeat() error = %v", err)
	}

	rec, _ := r.Get("w1")
	if rec.Status != "busy" {
		t.Errorf("Status = %q, want busy", rec.Status)
	}
	if rec.CPUUtilizationPercent == nil || *rec.CPUUtilizationPercent != 42.0 {
		t.Errorf("CPUUtilizationPercent = %v, want unchanged 42.0", rec.CPUUtilizationPercent)
	}
}

func TestWorkerRecord_ZeroValueHardwareSchemaVersionMeansLegacy(t *testing.T) {
	r := NewInMemoryRegistry()
	// A bare-minimum registration, as any Phase 1/2 test or caller
	// would send -- HardwareSchemaVersion left at its zero value.
	if _, err := r.Register(WorkerRecord{ID: "legacy", Address: "a", Hostname: "h"}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	rec, _ := r.Get("legacy")
	if rec.HardwareSchemaVersion != 0 {
		t.Errorf("HardwareSchemaVersion = %d, want 0 (meaning: treat as legacy/v1) for a record that never set it", rec.HardwareSchemaVersion)
	}
}
