package registry

import (
	"fmt"
	"math"
	"strings"
)

// ValidateHardware checks a worker's reported hardware profile for
// values that are impossible or internally inconsistent, and returns
// an error wrapping ErrInvalidHardware listing every problem found.
//
// It deliberately does NOT reject incomplete profiles: an unknown
// value (nil pointer, empty string, zero core count from a failed
// probe) is legitimate and stays unknown. Only contradictions are
// rejected -- e.g. more physical cores than logical processors, more
// RAM available than installed, a utilization of 140%.
//
// Rules are versioned: checks that schema-2 workers could not satisfy
// (e.g. schema 2 reported an integrated GPU's WMI aperture as
// memory_total_bytes) only apply from the schema version that
// introduced them, so older workers keep registering.
func ValidateHardware(rec WorkerRecord) error {
	var problems []string
	add := func(format string, args ...any) { problems = append(problems, fmt.Sprintf(format, args...)) }

	v := rec.HardwareSchemaVersion
	if v < 0 {
		add("hardware_schema_version %d is negative", v)
	}

	if rec.CPULogicalCores < 0 {
		add("cpu_logical_cores %d is negative", rec.CPULogicalCores)
	}
	if rec.CPUPhysicalCores < 0 {
		add("cpu_physical_cores %d is negative", rec.CPUPhysicalCores)
	}
	if rec.CPUPhysicalCores > 0 && rec.CPULogicalCores > 0 && rec.CPUPhysicalCores > rec.CPULogicalCores {
		add("cpu_physical_cores (%d) exceeds cpu_logical_cores (%d)", rec.CPUPhysicalCores, rec.CPULogicalCores)
	}
	if rec.TotalRAMBytes < 0 {
		add("total_ram_bytes %d is negative", rec.TotalRAMBytes)
	}
	checkPositiveFloat(add, "cpu_frequency_current_mhz", rec.CPUFrequencyCurrentMHz)
	checkPositiveFloat(add, "cpu_frequency_max_mhz", rec.CPUFrequencyMaxMHz)

	switch rec.ProfileSource {
	case "", ProfileSourceUnknown:
		if v >= 3 {
			add("hardware schema %d requires profile_source REAL or SIMULATED", v)
		}
	case ProfileSourceReal:
		if rec.SimulatedProfileName != "" {
			add("a REAL profile must not carry simulated_profile_name %q", rec.SimulatedProfileName)
		}
	case ProfileSourceSimulated:
		if rec.SimulatedProfileName == "" {
			add("a SIMULATED profile must name its simulated_profile_name")
		}
	default:
		add("unknown profile_source %q", rec.ProfileSource)
	}

	seenIDs := map[string]bool{}
	for i, g := range rec.GPUs {
		where := fmt.Sprintf("gpus[%d] (%s)", i, g.Name)
		switch g.MemoryType {
		case "", "dedicated", "shared", "unknown":
		default:
			add("%s: memory_type %q is not dedicated|shared|unknown", where, g.MemoryType)
		}
		checkNonNegativeInt(add, where+".memory_total_bytes", g.MemoryTotalBytes)
		checkNonNegativeInt(add, where+".dedicated_memory_bytes", g.DedicatedMemoryBytes)
		checkNonNegativeInt(add, where+".shared_memory_bytes", g.SharedMemoryBytes)
		if v >= 3 && g.MemoryType == "shared" && g.MemoryTotalBytes != nil {
			add("%s: an integrated (shared-memory) GPU must not report memory_total_bytes -- shared system RAM is not dedicated VRAM", where)
		}
		if g.ID != "" {
			if seenIDs[g.ID] {
				add("%s: duplicate gpu id %q", where, g.ID)
			}
			seenIDs[g.ID] = true
		}
	}

	for i, b := range rec.BackendCapabilities {
		where := fmt.Sprintf("backend_capabilities[%d]", i)
		if strings.TrimSpace(b.Name) == "" {
			add("%s: name is empty", where)
		}
		if b.DeviceCount < 0 {
			add("%s (%s): device_count %d is negative", where, b.Name, b.DeviceCount)
		}
		if b.TestPassed != nil && !b.Tested {
			add("%s (%s): test_passed is set but tested is false", where, b.Name)
		}
		checkPositiveFloat(add, where+".test_gflops", b.TestGFLOPS)
	}

	// The registration-time live metrics obey the same rules as
	// heartbeat metrics.
	problems = append(problems, liveMetricProblems(rec, LiveMetrics{
		CPUUtilizationPercent:  rec.CPUUtilizationPercent,
		RAMAvailableBytes:      rec.RAMAvailableBytes,
		RAMUsedBytes:           rec.RAMUsedBytes,
		RAMUtilizationPercent:  rec.RAMUtilizationPercent,
		CPUFrequencyCurrentMHz: rec.CPUFrequencyCurrentMHz,
		CPULoadAverage1m:       rec.CPULoadAverage1m,
		SwapTotalBytes:         rec.SwapTotalBytes,
		SwapUsedBytes:          rec.SwapUsedBytes,
		SwapUtilizationPercent: rec.SwapUtilizationPercent,
		GPUMetrics:             rec.GPUMetrics,
	})...)

	if len(problems) > 0 {
		return fmt.Errorf("%w: %s", ErrInvalidHardware, strings.Join(problems, "; "))
	}
	return nil
}

// ValidateLiveMetrics checks heartbeat metrics against the worker's
// stored (static) profile, returning an error wrapping
// ErrInvalidMetrics.
func ValidateLiveMetrics(rec WorkerRecord, m LiveMetrics) error {
	if problems := liveMetricProblems(rec, m); len(problems) > 0 {
		return fmt.Errorf("%w: %s", ErrInvalidMetrics, strings.Join(problems, "; "))
	}
	return nil
}

func liveMetricProblems(rec WorkerRecord, m LiveMetrics) []string {
	var problems []string
	add := func(format string, args ...any) { problems = append(problems, fmt.Sprintf(format, args...)) }

	checkPercent(add, "cpu_utilization_percent", m.CPUUtilizationPercent)
	checkPercent(add, "ram_utilization_percent", m.RAMUtilizationPercent)
	checkPercent(add, "swap_utilization_percent", m.SwapUtilizationPercent)
	checkNonNegativeInt(add, "ram_available_bytes", m.RAMAvailableBytes)
	checkNonNegativeInt(add, "ram_used_bytes", m.RAMUsedBytes)
	checkNonNegativeInt(add, "swap_total_bytes", m.SwapTotalBytes)
	checkNonNegativeInt(add, "swap_used_bytes", m.SwapUsedBytes)
	checkPositiveFloat(add, "cpu_frequency_current_mhz", m.CPUFrequencyCurrentMHz)
	if m.CPULoadAverage1m != nil && (*m.CPULoadAverage1m < 0 || math.IsNaN(*m.CPULoadAverage1m) || math.IsInf(*m.CPULoadAverage1m, 0)) {
		add("cpu_load_average_1m %v is not a non-negative number", *m.CPULoadAverage1m)
	}

	if total := rec.TotalRAMBytes; total > 0 {
		if m.RAMAvailableBytes != nil && *m.RAMAvailableBytes > total {
			add("ram_available_bytes (%d) exceeds total_ram_bytes (%d)", *m.RAMAvailableBytes, total)
		}
		if m.RAMUsedBytes != nil && *m.RAMUsedBytes > total {
			add("ram_used_bytes (%d) exceeds total_ram_bytes (%d)", *m.RAMUsedBytes, total)
		}
	}
	swapTotal := m.SwapTotalBytes
	if swapTotal == nil {
		swapTotal = rec.SwapTotalBytes
	}
	if swapTotal != nil && m.SwapUsedBytes != nil && *m.SwapUsedBytes > *swapTotal {
		add("swap_used_bytes (%d) exceeds swap_total_bytes (%d)", *m.SwapUsedBytes, *swapTotal)
	}

	known := map[string]GPUInfo{}
	for _, g := range rec.GPUs {
		if g.ID != "" {
			known[g.ID] = g
		}
	}
	for i, gm := range m.GPUMetrics {
		where := fmt.Sprintf("gpu_metrics[%d] (%s)", i, gm.GPUID)
		g, ok := known[gm.GPUID]
		if !ok {
			add("%s: gpu_id does not match any GPU in the registered profile", where)
			continue
		}
		checkPercent(add, where+".utilization_percent", gm.UtilizationPercent)
		checkNonNegativeInt(add, where+".dedicated_memory_used_bytes", gm.DedicatedMemoryUsedBytes)
		checkNonNegativeInt(add, where+".shared_memory_used_bytes", gm.SharedMemoryUsedBytes)
		if gm.DedicatedMemoryUsedBytes != nil && g.DedicatedMemoryBytes != nil && *gm.DedicatedMemoryUsedBytes > *g.DedicatedMemoryBytes {
			add("%s: dedicated_memory_used_bytes (%d) exceeds the GPU's dedicated_memory_bytes (%d)",
				where, *gm.DedicatedMemoryUsedBytes, *g.DedicatedMemoryBytes)
		}
	}
	return problems
}

func checkPercent(add func(string, ...any), name string, v *float64) {
	if v != nil && (math.IsNaN(*v) || *v < 0 || *v > 100) {
		add("%s %v is outside 0-100", name, *v)
	}
}

func checkNonNegativeInt(add func(string, ...any), name string, v *int64) {
	if v != nil && *v < 0 {
		add("%s %d is negative", name, *v)
	}
}

func checkPositiveFloat(add func(string, ...any), name string, v *float64) {
	if v != nil && (math.IsNaN(*v) || math.IsInf(*v, 0) || *v <= 0) {
		add("%s %v must be a positive number (omit it if unknown)", name, *v)
	}
}
