package registry

import (
	"errors"
	"strings"
	"testing"
	"time"
)

// f64 and i64 are shared with registry_hardware_test.go.
func bptr(v bool) *bool { return &v }

// computer1Like is a schema-3 REAL record shaped like Computer 1's
// actual profile (see docs/DEVELOPMENT_STATUS.md for the measured
// values this mirrors).
func computer1Like(id string) WorkerRecord {
	return WorkerRecord{
		ID:                    id,
		Address:               "127.0.0.1:50100",
		Hostname:              "YakobsPC",
		CPULogicalCores:       12,
		CPUPhysicalCores:      10,
		TotalRAMBytes:         16876888064,
		HardwareSchemaVersion: 3,
		ProfileSource:         ProfileSourceReal,
		CPUFrequencyMaxMHz:    f64(1700),
		GPUs: []GPUInfo{{
			ID: "luid:0x00000000_0x000160E5", Vendor: "Intel", Name: "Intel(R) Iris(R) Xe Graphics",
			MemoryType: "shared", Integrated: true, DedicatedMemoryBytes: i64(134217728), SharedMemoryBytes: i64(8438444032),
			MemorySource: "dxgi",
		}},
		TorchInstalled:      true,
		CPUBackendAvailable: true,
		BackendCapabilities: []BackendCapability{
			{Name: "cpu", Available: bptr(true), Tested: true, TestPassed: bptr(true), TestGFLOPS: f64(155.9)},
			{Name: "cuda", Available: bptr(false), Version: "12.6"},
		},
		RAMAvailableBytes: i64(3338182656),
		RAMUsedBytes:      i64(13538705408),
	}
}

func TestValidateHardware_AcceptsRealComputer1Profile(t *testing.T) {
	if err := ValidateHardware(computer1Like("c1")); err != nil {
		t.Fatalf("valid profile rejected: %v", err)
	}
}

func TestValidateHardware_AcceptsIncompleteProfile(t *testing.T) {
	// Unknown values are legitimate: a worker whose probes all failed.
	rec := WorkerRecord{ID: "w", HardwareSchemaVersion: 3, ProfileSource: ProfileSourceReal}
	if err := ValidateHardware(rec); err != nil {
		t.Fatalf("incomplete-but-consistent profile rejected: %v", err)
	}
}

func TestValidateHardware_RejectsImpossibleValues(t *testing.T) {
	cases := map[string]struct {
		mutate func(*WorkerRecord)
		want   string
	}{
		"physical > logical":   {func(r *WorkerRecord) { r.CPUPhysicalCores = 16 }, "exceeds cpu_logical_cores"},
		"negative cores":       {func(r *WorkerRecord) { r.CPULogicalCores = -1 }, "cpu_logical_cores -1 is negative"},
		"negative RAM":         {func(r *WorkerRecord) { r.TotalRAMBytes = -5 }, "total_ram_bytes -5 is negative"},
		"available > total":    {func(r *WorkerRecord) { r.RAMAvailableBytes = i64(r.TotalRAMBytes + 1) }, "ram_available_bytes"},
		"cpu util > 100":       {func(r *WorkerRecord) { r.CPUUtilizationPercent = f64(140) }, "cpu_utilization_percent 140"},
		"zero frequency":       {func(r *WorkerRecord) { r.CPUFrequencyMaxMHz = f64(0) }, "cpu_frequency_max_mhz"},
		"bad memory type":      {func(r *WorkerRecord) { r.GPUs[0].MemoryType = "magic" }, "memory_type \"magic\""},
		"shared reported VRAM": {func(r *WorkerRecord) { r.GPUs[0].MemoryTotalBytes = i64(2147479552) }, "shared system RAM is not dedicated VRAM"},
		"duplicate gpu id":     {func(r *WorkerRecord) { r.GPUs = append(r.GPUs, r.GPUs[0]) }, "duplicate gpu id"},
		"no source on schema3": {func(r *WorkerRecord) { r.ProfileSource = "" }, "requires profile_source"},
		"bogus source":         {func(r *WorkerRecord) { r.ProfileSource = "MAYBE" }, "unknown profile_source"},
		"simulated w/o name":   {func(r *WorkerRecord) { r.ProfileSource = ProfileSourceSimulated }, "must name its simulated_profile_name"},
		"real with sim name":   {func(r *WorkerRecord) { r.SimulatedProfileName = "x" }, "must not carry simulated_profile_name"},
		"untested but passed":  {func(r *WorkerRecord) { r.BackendCapabilities[1].TestPassed = bptr(true) }, "tested is false"},
		"empty backend name":   {func(r *WorkerRecord) { r.BackendCapabilities[0].Name = " " }, "name is empty"},
		"negative schema":      {func(r *WorkerRecord) { r.HardwareSchemaVersion = -1 }, "negative"},
		"gpu metric unknown":   {func(r *WorkerRecord) { r.GPUMetrics = []GPUMetrics{{GPUID: "nope"}} }, "does not match any GPU"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			rec := computer1Like("w")
			tc.mutate(&rec)
			err := ValidateHardware(rec)
			if !errors.Is(err, ErrInvalidHardware) {
				t.Fatalf("err = %v, want ErrInvalidHardware", err)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %q does not mention %q", err, tc.want)
			}
		})
	}
}

func TestValidateHardware_ListsEveryProblem(t *testing.T) {
	rec := computer1Like("w")
	rec.CPUPhysicalCores = 99
	rec.CPUUtilizationPercent = f64(-3)
	err := ValidateHardware(rec)
	if err == nil || !strings.Contains(err.Error(), "cpu_physical_cores") || !strings.Contains(err.Error(), "cpu_utilization_percent") {
		t.Fatalf("expected both problems listed, got %v", err)
	}
}

func TestValidateHardware_VersionedRules(t *testing.T) {
	// Schema 2 workers reported an integrated GPU's WMI aperture as
	// memory_total_bytes; they must keep registering (backward compat).
	rec := computer1Like("w")
	rec.HardwareSchemaVersion = 2
	rec.ProfileSource = ""
	rec.GPUs[0].MemoryTotalBytes = i64(2147479552)
	if err := ValidateHardware(rec); err != nil {
		t.Fatalf("schema 2 profile rejected by a schema-3-only rule: %v", err)
	}
	// A newer worker than this orchestrator knows is accepted too.
	rec = computer1Like("w")
	rec.HardwareSchemaVersion = LatestHardwareSchemaVersion + 1
	if err := ValidateHardware(rec); err != nil {
		t.Fatalf("newer schema rejected: %v", err)
	}
}

func TestRegister_RejectsInvalidHardwareAndStoresNothing(t *testing.T) {
	r := NewInMemoryRegistry()
	rec := computer1Like("bad")
	rec.CPUPhysicalCores = 64
	if _, err := r.Register(rec); !errors.Is(err, ErrInvalidHardware) {
		t.Fatalf("err = %v, want ErrInvalidHardware", err)
	}
	if _, ok := r.Get("bad"); ok {
		t.Fatal("invalid registration was stored")
	}
}

func TestRegister_ProfileSourceDefaultsToUnknownNotReal(t *testing.T) {
	r := NewInMemoryRegistry()
	if _, err := r.Register(WorkerRecord{ID: "old", Address: "a"}); err != nil {
		t.Fatal(err)
	}
	rec, _ := r.Get("old")
	if rec.ProfileSource != ProfileSourceUnknown {
		t.Errorf("ProfileSource = %q, want UNKNOWN (never assume REAL)", rec.ProfileSource)
	}
}

func TestRegister_SetsHardwareTimestamps(t *testing.T) {
	r := NewInMemoryRegistry()
	collected := time.Now().Add(-2 * time.Second).Truncate(time.Millisecond)
	rec := computer1Like("w")
	rec.ProfileCollectedAt = collected
	before := time.Now()
	if _, err := r.Register(rec); err != nil {
		t.Fatal(err)
	}
	got, _ := r.Get("w")
	if got.HardwareUpdatedAt.Before(before) || got.MetricsUpdatedAt.Before(before) {
		t.Errorf("timestamps not set by registry: hw=%v metrics=%v", got.HardwareUpdatedAt, got.MetricsUpdatedAt)
	}
	if !got.ProfileCollectedAt.Equal(collected) || !got.MetricsCollectedAt.Equal(collected) {
		t.Errorf("worker collection time not preserved: %v / %v", got.ProfileCollectedAt, got.MetricsCollectedAt)
	}
}

func TestHeartbeat_UnknownWorkerIsErrWorkerNotFound(t *testing.T) {
	r := NewInMemoryRegistry()
	if err := r.Heartbeat("ghost", "idle", time.Now()); !errors.Is(err, ErrWorkerNotFound) {
		t.Fatalf("err = %v, want ErrWorkerNotFound", err)
	}
	if err := r.Deregister("ghost"); !errors.Is(err, ErrWorkerNotFound) {
		t.Fatalf("Deregister err = %v, want ErrWorkerNotFound", err)
	}
}

func TestHeartbeat_InvalidMetricsRejectedRecordUnchanged(t *testing.T) {
	r := NewInMemoryRegistry()
	if _, err := r.Register(computer1Like("w")); err != nil {
		t.Fatal(err)
	}
	before, _ := r.Get("w")
	cases := []LiveMetrics{
		{CPUUtilizationPercent: f64(101)},
		{RAMAvailableBytes: i64(before.TotalRAMBytes * 2)},
		{SwapTotalBytes: i64(10), SwapUsedBytes: i64(11)},
		{GPUMetrics: []GPUMetrics{{GPUID: "not-registered", UtilizationPercent: f64(5)}}},
		{GPUMetrics: []GPUMetrics{{GPUID: "luid:0x00000000_0x000160E5", UtilizationPercent: f64(250)}}},
		{GPUMetrics: []GPUMetrics{{GPUID: "luid:0x00000000_0x000160E5", DedicatedMemoryUsedBytes: i64(1 << 40)}}},
		{CPULoadAverage1m: f64(-1)},
	}
	for i, m := range cases {
		err := r.HeartbeatWithMetrics("w", "busy", time.Now(), m)
		if !errors.Is(err, ErrInvalidMetrics) {
			t.Errorf("case %d: err = %v, want ErrInvalidMetrics", i, err)
		}
	}
	after, _ := r.Get("w")
	if after.Status != before.Status || !after.LastHeartbeat.Equal(before.LastHeartbeat) {
		t.Error("a rejected heartbeat must not change the stored record")
	}
}

func TestHeartbeat_Schema3MetricsUpdateAndGPUMerge(t *testing.T) {
	r := NewInMemoryRegistry()
	if _, err := r.Register(computer1Like("w")); err != nil {
		t.Fatal(err)
	}
	gpuID := "luid:0x00000000_0x000160E5"
	collected := time.Now().Truncate(time.Millisecond)
	if err := r.HeartbeatWithMetrics("w", "idle", time.Now(), LiveMetrics{
		CPUFrequencyCurrentMHz: f64(1700),
		SwapTotalBytes:         i64(14 << 30),
		SwapUsedBytes:          i64(2 << 30),
		SwapUtilizationPercent: f64(15.7),
		GPUMetrics:             []GPUMetrics{{GPUID: gpuID, UtilizationPercent: f64(5.2), SharedMemoryUsedBytes: i64(1248112640), DedicatedMemoryUsedBytes: i64(0)}},
		CollectedAt:            collected,
	}); err != nil {
		t.Fatal(err)
	}
	first, _ := r.Get("w")
	// A later beat with only utilization keeps the stored memory figures.
	if err := r.HeartbeatWithMetrics("w", "idle", time.Now(), LiveMetrics{
		GPUMetrics: []GPUMetrics{{GPUID: gpuID, UtilizationPercent: f64(40)}},
	}); err != nil {
		t.Fatal(err)
	}
	// A beat with no GPU metrics at all keeps everything.
	if err := r.Heartbeat("w", "idle", time.Now()); err != nil {
		t.Fatal(err)
	}
	got, _ := r.Get("w")
	if len(got.GPUMetrics) != 1 {
		t.Fatalf("GPUMetrics = %+v, want one merged entry", got.GPUMetrics)
	}
	m := got.GPUMetrics[0]
	if *m.UtilizationPercent != 40 || *m.SharedMemoryUsedBytes != 1248112640 || *m.DedicatedMemoryUsedBytes != 0 {
		t.Errorf("merge wrong: util=%v shared=%v ded=%v", *m.UtilizationPercent, *m.SharedMemoryUsedBytes, *m.DedicatedMemoryUsedBytes)
	}
	if *got.SwapUsedBytes != 2<<30 || *got.CPUFrequencyCurrentMHz != 1700 {
		t.Error("schema 3 scalar metrics not stored")
	}
	if !first.MetricsCollectedAt.Equal(collected) {
		t.Errorf("MetricsCollectedAt = %v, want %v", first.MetricsCollectedAt, collected)
	}
	// The record handed out earlier must not have been mutated in place.
	if *first.GPUMetrics[0].UtilizationPercent != 5.2 {
		t.Error("Get() result aliased registry storage")
	}
}

func TestHeartbeat_MetricsUpdatedAtOnlyMovesWhenMetricsChange(t *testing.T) {
	r := NewInMemoryRegistry()
	if _, err := r.Register(computer1Like("w")); err != nil {
		t.Fatal(err)
	}
	reg, _ := r.Get("w")
	later := reg.MetricsUpdatedAt.Add(5 * time.Second)
	if err := r.Heartbeat("w", "idle", later); err != nil {
		t.Fatal(err)
	}
	got, _ := r.Get("w")
	if !got.MetricsUpdatedAt.Equal(reg.MetricsUpdatedAt) {
		t.Error("a metric-less heartbeat must not claim the metrics are fresh")
	}
	if err := r.HeartbeatWithMetrics("w", "idle", later, LiveMetrics{CPUUtilizationPercent: f64(9)}); err != nil {
		t.Fatal(err)
	}
	got, _ = r.Get("w")
	if !got.MetricsUpdatedAt.Equal(later) {
		t.Errorf("MetricsUpdatedAt = %v, want %v", got.MetricsUpdatedAt, later)
	}
}

func TestReconnection_ReRegisterRefreshesHardware(t *testing.T) {
	r := NewInMemoryRegistryWithTimeout(50 * time.Millisecond)
	if _, err := r.Register(computer1Like("w")); err != nil {
		t.Fatal(err)
	}
	time.Sleep(80 * time.Millisecond) // goes offline
	if rec, _ := r.Get("w"); rec.Online {
		t.Fatal("expected offline")
	}
	fresh := computer1Like("w")
	fresh.Address = "192.168.1.50:50100" // reconnect from a LAN address
	fresh.RAMAvailableBytes = i64(8 << 30)
	if isNew, err := r.Register(fresh); err != nil || isNew {
		t.Fatalf("reconnect: isNew=%v err=%v", isNew, err)
	}
	rec, _ := r.Get("w")
	if !rec.Online || rec.Address != "192.168.1.50:50100" || *rec.RAMAvailableBytes != 8<<30 {
		t.Errorf("reconnection did not refresh the profile: %+v", rec)
	}
}
