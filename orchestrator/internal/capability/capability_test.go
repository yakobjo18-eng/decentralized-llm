package capability

import (
	"strings"
	"testing"
	"time"

	"orchestrator/internal/registry"
)

func f64(v float64) *float64 { return &v }
func i64(v int64) *int64     { return &v }
func bptr(v bool) *bool      { return &v }

// computer1 mirrors Computer 1's measured schema-3 profile.
func computer1() registry.WorkerRecord {
	return registry.WorkerRecord{
		ID: "computer-1", Online: true, ProfileSource: registry.ProfileSourceReal, HardwareSchemaVersion: 3,
		CPULogicalCores: 12, CPUPhysicalCores: 10, TotalRAMBytes: 16876888064,
		CPUFrequencyMaxMHz: f64(1700), CPUUtilizationPercent: f64(14.6),
		RAMAvailableBytes: i64(3338182656), TorchInstalled: true, CPUBackendAvailable: true,
		GPUs: []registry.GPUInfo{{
			ID: "luid:A", Vendor: "Intel", Name: "Intel(R) Iris(R) Xe Graphics", MemoryType: "shared", Integrated: true,
			DedicatedMemoryBytes: i64(134217728), SharedMemoryBytes: i64(8438444032),
		}},
		GPUMetrics: []registry.GPUMetrics{{GPUID: "luid:A", UtilizationPercent: f64(5.2), DedicatedMemoryUsedBytes: i64(0)}},
		BackendCapabilities: []registry.BackendCapability{
			{Name: "cpu", Available: bptr(true), Tested: true, TestPassed: bptr(true), TestGFLOPS: f64(155.89)},
			{Name: "cuda", Available: bptr(false), Version: "12.6", Detail: "torch built for CUDA 12.6, but torch.cuda.is_available() is False"},
			{Name: "xpu", Available: bptr(false)},
		},
	}
}

func nvidiaWorker(cudaUsable bool) registry.WorkerRecord {
	rec := computer1()
	rec.ID = "gpu-box"
	rec.TotalRAMBytes = 32 << 30
	rec.RAMAvailableBytes = i64(24 << 30)
	rec.GPUs = []registry.GPUInfo{{
		ID: "luid:N", Vendor: "NVIDIA", Name: "RTX", MemoryType: "dedicated",
		DedicatedMemoryBytes: i64(8 << 30), MemoryTotalBytes: i64(8 << 30), Backends: []string{"CUDA"},
	}}
	rec.GPUMetrics = []registry.GPUMetrics{{GPUID: "luid:N", DedicatedMemoryUsedBytes: i64(1 << 30)}}
	rec.BackendCapabilities = []registry.BackendCapability{
		{Name: "cpu", Available: bptr(true), Tested: true, TestPassed: bptr(true)},
		{Name: "cuda", Available: bptr(cudaUsable), Tested: cudaUsable, TestPassed: map[bool]*bool{true: bptr(true), false: nil}[cudaUsable], DeviceCount: 1},
	}
	return rec
}

func TestClassify_Computer1IsCPUOnlyStandardMemory(t *testing.T) {
	c := Classify(computer1(), DefaultConfig())
	if !c.Has(CPUOnly) || c.Has(GPUAccelerated) {
		t.Fatalf("classes = %v, want CPU_ONLY (Iris Xe present but has no compute backend)", c.Classes)
	}
	if c.MemoryTier != "STANDARD" || c.Has(HighMemory) || c.Has(LowMemory) {
		t.Errorf("15.7 GiB should be STANDARD with defaults, got tier %s classes %v", c.MemoryTier, c.Classes)
	}
	if c.Has(MixedHardware) {
		t.Error("one GPU vendor, no accelerators: not mixed")
	}
	joined := strings.Join(c.Reasons, " | ")
	for _, want := range []string{"Iris(R) Xe", "no usable compute backend", "cuda unavailable", "15.72 GiB"} {
		if !strings.Contains(joined, want) {
			t.Errorf("reasons %q missing %q", joined, want)
		}
	}
}

func TestClassify_GPUPresenceIsNotGPUCapability(t *testing.T) {
	// An NVIDIA card detected, but torch can't use it (e.g. CPU-only
	// torch build, or a broken driver): still CPU_ONLY.
	c := Classify(nvidiaWorker(false), DefaultConfig())
	if !c.Has(CPUOnly) || c.Has(GPUAccelerated) {
		t.Fatalf("classes = %v, want CPU_ONLY", c.Classes)
	}
	c = Classify(nvidiaWorker(true), DefaultConfig())
	if !c.Has(GPUAccelerated) || c.Has(CPUOnly) {
		t.Fatalf("classes = %v, want GPU_ACCELERATED", c.Classes)
	}
	if strings.Join(c.AcceleratorBackends, ",") != "cuda" {
		t.Errorf("accelerators = %v", c.AcceleratorBackends)
	}
	if !c.Has(HighMemory) {
		t.Errorf("32 GiB should be HIGH_MEMORY with defaults, got %v", c.Classes)
	}
}

func TestClassify_FailedComputeTestIsNotUsable(t *testing.T) {
	rec := nvidiaWorker(true)
	rec.BackendCapabilities[1].TestPassed = bptr(false)
	if c := Classify(rec, DefaultConfig()); c.Has(GPUAccelerated) {
		t.Fatalf("a backend that failed its compute test must not count: %v", c.Classes)
	}
}

func TestClassify_LowMemoryMixedNoBackendUnclassified(t *testing.T) {
	rec := computer1()
	rec.TotalRAMBytes = 4 << 30
	if c := Classify(rec, DefaultConfig()); !c.Has(LowMemory) || c.MemoryTier != "LOW" {
		t.Errorf("4 GiB: %v / %s", c.Classes, c.MemoryTier)
	}

	mixed := nvidiaWorker(true)
	mixed.GPUs = append(mixed.GPUs, computer1().GPUs[0]) // NVIDIA + Intel iGPU laptop
	if c := Classify(mixed, DefaultConfig()); !c.Has(MixedHardware) || !c.Has(GPUAccelerated) {
		t.Errorf("NVIDIA + Intel: %v", c.Classes)
	}

	none := computer1()
	none.TorchInstalled = false
	none.CPUBackendAvailable = false
	none.BackendCapabilities = []registry.BackendCapability{{Name: "cpu"}}
	if c := Classify(none, DefaultConfig()); !c.Has(NoComputeBackend) {
		t.Errorf("no torch: %v", c.Classes)
	}

	old := registry.WorkerRecord{ID: "phase1", TotalRAMBytes: 0}
	c := Classify(old, DefaultConfig())
	if !c.Has(Unclassified) || c.MemoryTier != "UNKNOWN" {
		t.Errorf("schema 1 record: %v / %s", c.Classes, c.MemoryTier)
	}
}

func TestClassify_PendingBackendProbeIsNotNoBackend(t *testing.T) {
	rec := computer1()
	rec.BackendProbePending = true
	rec.TorchInstalled = false // meaningless while pending
	rec.CPUBackendAvailable = false
	rec.BackendCapabilities = nil
	c := Classify(rec, DefaultConfig())
	if !c.Has(Unclassified) || c.Has(NoComputeBackend) || c.Has(CPUOnly) {
		t.Fatalf("pending probe: %v", c.Classes)
	}
	if c.MemoryTier != "STANDARD" {
		t.Errorf("memory tier should still be derived: %s", c.MemoryTier)
	}
	res := ComputeResources(rec, DefaultConfig(), time.Now())
	if !res.BackendProbePending || len(res.ExecutionBackends) != 0 {
		t.Errorf("resources while pending: %+v", res)
	}
}

func TestClassify_Schema2FallsBackToCUDAFlag(t *testing.T) {
	rec := registry.WorkerRecord{HardwareSchemaVersion: 2, TotalRAMBytes: 16 << 30, TorchInstalled: true, CPUBackendAvailable: true, CUDAAvailable: bptr(true)}
	if c := Classify(rec, DefaultConfig()); !c.Has(GPUAccelerated) {
		t.Errorf("schema 2 with cuda_available=true: %v", c.Classes)
	}
}

func TestClassify_ThresholdsAreConfigurable(t *testing.T) {
	cfg := DefaultConfig()
	cfg.HighMemoryThresholdGiB = 12
	if c := Classify(computer1(), cfg); !c.Has(HighMemory) {
		t.Errorf("with a 12 GiB threshold Computer 1 should be HIGH_MEMORY: %v", c.Classes)
	}
	cfg = DefaultConfig()
	cfg.LowMemoryThresholdGiB = 16
	if c := Classify(computer1(), cfg); !c.Has(LowMemory) {
		t.Errorf("with a 16 GiB low threshold Computer 1 should be LOW_MEMORY: %v", c.Classes)
	}
}

func TestConfigValidate(t *testing.T) {
	if err := DefaultConfig().Validate(); err != nil {
		t.Fatal(err)
	}
	bad := DefaultConfig()
	bad.LowMemoryThresholdGiB = 64
	if bad.Validate() == nil {
		t.Error("low >= high must be rejected")
	}
	bad = DefaultConfig()
	bad.RAMSafetyMarginPercent = 100
	if bad.Validate() == nil {
		t.Error("100% margin must be rejected")
	}
}

func TestResources_Computer1UsesAvailableRAMNotTotal(t *testing.T) {
	now := time.Now()
	rec := computer1()
	rec.MetricsUpdatedAt = now.Add(-3 * time.Second)
	res := ComputeResources(rec, DefaultConfig(), now)

	total, pct := float64(16876888064), 0.10
	wantMargin := int64(total * pct) // 10% (1.57 GiB) > 1 GiB floor
	if res.RAMSafetyMarginBytes != wantMargin {
		t.Errorf("margin = %d, want %d", res.RAMSafetyMarginBytes, wantMargin)
	}
	if res.RAMUsableBytes == nil || *res.RAMUsableBytes != 3338182656-wantMargin {
		t.Errorf("usable = %v, want available - margin", res.RAMUsableBytes)
	}
	if strings.Join(res.ExecutionBackends, ",") != "cpu" {
		t.Errorf("backends = %v", res.ExecutionBackends)
	}
	if res.CPUMeasuredGFLOPS == nil || *res.CPUMeasuredGFLOPS != 155.89 {
		t.Errorf("cpu gflops = %v", res.CPUMeasuredGFLOPS)
	}
	if len(res.GPUs) != 1 || res.GPUs[0].VRAMTotalBytes != nil || res.GPUs[0].ComputeUsable {
		t.Errorf("iGPU must have no VRAM and not be compute-usable: %+v", res.GPUs)
	}
	if res.TotalUsableVRAMBytes != nil {
		t.Error("no compute-usable GPU: total usable VRAM must be unknown/absent")
	}
	if res.MetricsAgeSeconds == nil || *res.MetricsAgeSeconds < 2.9 {
		t.Errorf("metrics age = %v", res.MetricsAgeSeconds)
	}
}

func TestResources_RAMEdgeCases(t *testing.T) {
	rec := computer1()
	rec.RAMAvailableBytes = nil
	if res := ComputeResources(rec, DefaultConfig(), time.Now()); res.RAMUsableBytes != nil {
		t.Error("unknown available RAM must give unknown usable RAM, not total")
	}
	rec.RAMAvailableBytes = i64(100 << 20) // 100 MiB free, below the margin
	if res := ComputeResources(rec, DefaultConfig(), time.Now()); res.RAMUsableBytes == nil || *res.RAMUsableBytes != 0 {
		t.Errorf("usable must floor at 0, got %v", res.RAMUsableBytes)
	}
	small := computer1()
	small.TotalRAMBytes = 4 << 30
	if res := ComputeResources(small, DefaultConfig(), time.Now()); res.RAMSafetyMarginBytes != 1<<30 {
		t.Errorf("margin floor not applied: %d", res.RAMSafetyMarginBytes)
	}
}

func TestResources_DiscreteGPUUsableVRAM(t *testing.T) {
	res := ComputeResources(nvidiaWorker(true), DefaultConfig(), time.Now())
	g := res.GPUs[0]
	vram := float64(8 << 30)
	margin := int64(vram * 0.10) // 10% of 8 GiB = 0.8 GiB > 0.5 GiB floor
	if g.VRAMUsableBytes == nil || *g.VRAMUsableBytes != (8<<30)-(1<<30)-margin {
		t.Errorf("usable VRAM = %v", g.VRAMUsableBytes)
	}
	if !g.ComputeUsable || res.TotalUsableVRAMBytes == nil || *res.TotalUsableVRAMBytes != *g.VRAMUsableBytes {
		t.Errorf("total usable VRAM = %v", res.TotalUsableVRAMBytes)
	}
	if strings.Join(res.ExecutionBackends, ",") != "cpu,cuda" {
		t.Errorf("backends = %v", res.ExecutionBackends)
	}

	// Detected but not usable: VRAM figures shown, but not counted.
	res = ComputeResources(nvidiaWorker(false), DefaultConfig(), time.Now())
	if res.GPUs[0].ComputeUsable || res.TotalUsableVRAMBytes != nil {
		t.Error("an unusable GPU's VRAM must not be counted as usable")
	}

	// Unknown VRAM usage -> unknown usable VRAM, never assumed free.
	rec := nvidiaWorker(true)
	rec.GPUMetrics = nil
	res = ComputeResources(rec, DefaultConfig(), time.Now())
	if res.GPUs[0].VRAMUsableBytes != nil || res.TotalUsableVRAMBytes != nil {
		t.Error("VRAM in use unknown: usable must be unknown")
	}
}
