// Package capability derives coarse capability classes and
// partition-planning resource figures from the hardware a worker
// actually reported (internal/registry). It never adds information:
// every class and every number here is computed from reported fields,
// with the reasoning recorded alongside, and anything that depends on
// an unreported value is left unknown (nil) rather than assumed.
//
// Classes are labels for humans and coarse filtering. They are NOT a
// substitute for the numbers: the Phase 4 partitioner is expected to
// plan with Resources (usable RAM/VRAM after safety margins, measured
// throughput), not with class names. Thresholds are configurable
// (orchestrator.yaml "capability:" section) so the classification can
// evolve with the partitioner without code changes.
package capability

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"orchestrator/internal/registry"
)

// Class is one capability label. A worker can carry several.
type Class string

const (
	// CPUOnly: a CPU inference backend works and no accelerator backend
	// does. A GPU may still be physically present (e.g. Computer 1's
	// Iris Xe) -- presence is not usability.
	CPUOnly Class = "CPU_ONLY"
	// GPUAccelerated: at least one non-CPU backend (cuda, xpu, rocm,
	// directml, mps, ...) is available and did not fail its compute test.
	GPUAccelerated Class = "GPU_ACCELERATED"
	// HighMemory / LowMemory: installed RAM at/above or below the
	// configured thresholds.
	HighMemory Class = "HIGH_MEMORY"
	LowMemory  Class = "LOW_MEMORY"
	// MixedHardware: GPUs from more than one vendor, or more than one
	// usable accelerator backend, on the same worker.
	MixedHardware Class = "MIXED_HARDWARE"
	// NoComputeBackend: the worker reported its backends and none works
	// (e.g. PyTorch not installed). It can't run inference yet.
	NoComputeBackend Class = "NO_COMPUTE_BACKEND"
	// Unclassified: the worker's hardware schema predates backend
	// reporting, so there is nothing to classify from.
	Unclassified Class = "UNCLASSIFIED"
)

const gib = 1 << 30

// Config holds every tunable. Plain scalars only, so
// orchestrator/internal/config.Config stays comparable with ==.
type Config struct {
	// Installed RAM at or above this is HIGH_MEMORY.
	HighMemoryThresholdGiB float64 `yaml:"high_memory_threshold_gib"`
	// Installed RAM below this is LOW_MEMORY.
	LowMemoryThresholdGiB float64 `yaml:"low_memory_threshold_gib"`

	// Memory a partition plan must leave free on a worker, so the OS
	// and the worker process itself aren't starved: the larger of a
	// percentage of the total and a fixed floor.
	RAMSafetyMarginPercent  float64 `yaml:"ram_safety_margin_percent"`
	RAMSafetyMarginMinGiB   float64 `yaml:"ram_safety_margin_min_gib"`
	VRAMSafetyMarginPercent float64 `yaml:"vram_safety_margin_percent"`
	VRAMSafetyMarginMinGiB  float64 `yaml:"vram_safety_margin_min_gib"`
}

// DefaultConfig: 8/32 GiB memory tiers (Computer 1, at 15.7 GiB, is
// neither), 10% / >=1 GiB RAM margin, 10% / >=0.5 GiB VRAM margin.
// Starting points for Phase 4 to tune with real model sizes, not
// derived from any measurement.
func DefaultConfig() Config {
	return Config{
		HighMemoryThresholdGiB:  32,
		LowMemoryThresholdGiB:   8,
		RAMSafetyMarginPercent:  10,
		RAMSafetyMarginMinGiB:   1,
		VRAMSafetyMarginPercent: 10,
		VRAMSafetyMarginMinGiB:  0.5,
	}
}

// Validate rejects configurations that would make classification
// meaningless.
func (c Config) Validate() error {
	var problems []string
	if c.LowMemoryThresholdGiB < 0 || c.HighMemoryThresholdGiB <= 0 {
		problems = append(problems, "memory thresholds must be positive")
	}
	if c.LowMemoryThresholdGiB >= c.HighMemoryThresholdGiB {
		problems = append(problems, "low_memory_threshold_gib must be below high_memory_threshold_gib")
	}
	for name, pct := range map[string]float64{"ram_safety_margin_percent": c.RAMSafetyMarginPercent, "vram_safety_margin_percent": c.VRAMSafetyMarginPercent} {
		if pct < 0 || pct >= 100 {
			problems = append(problems, name+" must be in [0, 100)")
		}
	}
	if c.RAMSafetyMarginMinGiB < 0 || c.VRAMSafetyMarginMinGiB < 0 {
		problems = append(problems, "safety margin minimums must not be negative")
	}
	if len(problems) > 0 {
		sort.Strings(problems)
		return fmt.Errorf("invalid capability config: %s", strings.Join(problems, "; "))
	}
	return nil
}

// Classification is the result of Classify.
type Classification struct {
	Classes []Class `json:"classes"`
	// "LOW" | "STANDARD" | "HIGH" | "UNKNOWN".
	MemoryTier string `json:"memory_tier"`
	// Usable accelerator backend names, sorted (e.g. ["cuda"]).
	AcceleratorBackends []string `json:"accelerator_backends"`
	// One sentence per decision, quoting the reported values used.
	Reasons []string `json:"reasons"`
}

// Has reports whether c includes class.
func (c Classification) Has(class Class) bool {
	for _, x := range c.Classes {
		if x == class {
			return true
		}
	}
	return false
}

// usableAccelerators returns the usable non-CPU backends. Schema 3
// workers report per-backend results; schema 2 workers only had the
// cuda/directml flags, which are used as a fallback.
func usableAccelerators(rec registry.WorkerRecord) []string {
	set := map[string]bool{}
	if len(rec.BackendCapabilities) > 0 {
		for _, b := range rec.BackendCapabilities {
			if b.Name != "cpu" && b.Usable() {
				set[strings.ToLower(b.Name)] = true
			}
		}
	} else {
		if rec.CUDAAvailable != nil && *rec.CUDAAvailable {
			set["cuda"] = true
		}
		if rec.DirectMLAvailable != nil && *rec.DirectMLAvailable {
			set["directml"] = true
		}
	}
	out := make([]string, 0, len(set))
	for name := range set {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

func cpuUsable(rec registry.WorkerRecord) bool {
	for _, b := range rec.BackendCapabilities {
		if b.Name == "cpu" {
			return b.Usable()
		}
	}
	return rec.CPUBackendAvailable
}

// Classify labels a worker from its reported profile.
func Classify(rec registry.WorkerRecord, cfg Config) Classification {
	out := Classification{AcceleratorBackends: usableAccelerators(rec)}
	add := func(c Class, reason string, args ...any) {
		out.Classes = append(out.Classes, c)
		out.Reasons = append(out.Reasons, fmt.Sprintf("%s: %s", c, fmt.Sprintf(reason, args...)))
	}
	note := func(reason string, args ...any) { out.Reasons = append(out.Reasons, fmt.Sprintf(reason, args...)) }

	schema := rec.HardwareSchemaVersion
	if schema == 0 {
		schema = 1
	}

	switch {
	case rec.BackendProbePending:
		add(Unclassified, "the worker's inference backend probe is still running; compute classes will follow its next registration")
	case len(out.AcceleratorBackends) > 0:
		add(GPUAccelerated, "usable accelerator backend(s): %s", strings.Join(out.AcceleratorBackends, ", "))
	case cpuUsable(rec):
		add(CPUOnly, "CPU backend works and no accelerator backend is usable")
		for _, b := range rec.BackendCapabilities {
			if b.Name != "cpu" && b.Detail != "" {
				note("%s unavailable: %s", b.Name, b.Detail)
			}
		}
	case schema < 2:
		add(Unclassified, "hardware schema %d predates backend reporting", schema)
	default:
		add(NoComputeBackend, "no inference backend reported usable (torch_installed=%t)", rec.TorchInstalled)
	}
	for _, g := range rec.GPUs {
		if !rec.BackendProbePending && !gpuComputeUsable(g, out.AcceleratorBackends) {
			note("GPU %q (%s) is present but no usable compute backend targets it", g.Name, g.Vendor)
		}
	}

	total := float64(rec.TotalRAMBytes) / gib
	switch {
	case rec.TotalRAMBytes <= 0:
		out.MemoryTier = "UNKNOWN"
		note("total RAM not reported; memory tier unknown")
	case total >= cfg.HighMemoryThresholdGiB:
		out.MemoryTier = "HIGH"
		add(HighMemory, "%.2f GiB installed >= %.2f GiB threshold", total, cfg.HighMemoryThresholdGiB)
	case total < cfg.LowMemoryThresholdGiB:
		out.MemoryTier = "LOW"
		add(LowMemory, "%.2f GiB installed < %.2f GiB threshold", total, cfg.LowMemoryThresholdGiB)
	default:
		out.MemoryTier = "STANDARD"
		note("memory tier STANDARD: %.2f GiB installed, between %.2f and %.2f GiB thresholds",
			total, cfg.LowMemoryThresholdGiB, cfg.HighMemoryThresholdGiB)
	}

	vendors := map[string]bool{}
	for _, g := range rec.GPUs {
		if g.Vendor != "" && g.Vendor != "Unknown" {
			vendors[g.Vendor] = true
		}
	}
	if len(vendors) > 1 || len(out.AcceleratorBackends) > 1 {
		names := make([]string, 0, len(vendors))
		for v := range vendors {
			names = append(names, v)
		}
		sort.Strings(names)
		add(MixedHardware, "GPU vendors [%s], usable accelerator backends [%s]",
			strings.Join(names, ", "), strings.Join(out.AcceleratorBackends, ", "))
	}
	return out
}

// backendVendors maps an accelerator backend to the GPU vendors it can
// drive. directml/vulkan run on any vendor's GPU.
var backendVendors = map[string][]string{
	"cuda": {"NVIDIA"},
	"rocm": {"AMD"},
	"hip":  {"AMD"},
	"xpu":  {"Intel"},
	"mps":  {"Apple"},
}

func gpuComputeUsable(g registry.GPUInfo, accelerators []string) bool {
	for _, a := range accelerators {
		vendors, specific := backendVendors[a]
		if !specific {
			return true // directml, vulkan, future vendor-neutral backends
		}
		for _, v := range vendors {
			if v == g.Vendor {
				return true
			}
		}
	}
	return false
}

// GPUResources is one GPU's partition-planning view.
type GPUResources struct {
	GPUID         string `json:"gpu_id"`
	Name          string `json:"name"`
	Vendor        string `json:"vendor"`
	MemoryType    string `json:"memory_type"`
	ComputeUsable bool   `json:"compute_usable"`
	// Dedicated VRAM only. nil for integrated GPUs (their memory IS
	// system RAM, already counted under RAM) and whenever unreported.
	VRAMTotalBytes        *int64   `json:"vram_total_bytes,omitempty"`
	VRAMUsedBytes         *int64   `json:"vram_used_bytes,omitempty"`
	VRAMSafetyMarginBytes *int64   `json:"vram_safety_margin_bytes,omitempty"`
	VRAMUsableBytes       *int64   `json:"vram_usable_bytes,omitempty"`
	UtilizationPercent    *float64 `json:"utilization_percent,omitempty"`
	Note                  string   `json:"note,omitempty"`
}

// Resources is what a partitioning engine needs from one worker.
// Usable figures are AVAILABLE memory minus the safety margin -- never
// installed totals.
type Resources struct {
	WorkerID      string `json:"worker_id"`
	Online        bool   `json:"online"`
	ProfileSource string `json:"profile_source"`

	RAMTotalBytes        int64  `json:"ram_total_bytes"`
	RAMAvailableBytes    *int64 `json:"ram_available_bytes,omitempty"`
	RAMSafetyMarginBytes int64  `json:"ram_safety_margin_bytes"`
	// max(0, available - margin); nil when available is unreported.
	RAMUsableBytes *int64 `json:"ram_usable_bytes,omitempty"`
	// Page-file pressure, reported as-is.
	SwapUtilizationPercent *float64 `json:"swap_utilization_percent,omitempty"`

	CPUPhysicalCores      int32    `json:"cpu_physical_cores"`
	CPULogicalCores       int32    `json:"cpu_logical_cores"`
	CPUUtilizationPercent *float64 `json:"cpu_utilization_percent,omitempty"`
	CPUFrequencyMaxMHz    *float64 `json:"cpu_frequency_max_mhz,omitempty"`
	// From the worker's own verified matmul micro-test; nil if not run.
	CPUMeasuredGFLOPS *float64 `json:"cpu_measured_gflops,omitempty"`

	// Usable execution backends, "cpu" first.
	ExecutionBackends []string `json:"execution_backends"`
	// Per-backend micro-test throughput (GFLOPS), where measured.
	BackendGFLOPS map[string]float64 `json:"backend_gflops"`
	// True while the worker's backend probe hasn't reported yet:
	// ExecutionBackends is then empty because it is unknown, not
	// because nothing works.
	BackendProbePending bool `json:"backend_probe_pending"`

	GPUs []GPUResources `json:"gpus"`
	// Sum of VRAMUsableBytes over compute-usable GPUs; nil if none of
	// them has a known usable figure.
	TotalUsableVRAMBytes *int64 `json:"total_usable_vram_bytes,omitempty"`

	MetricsAgeSeconds *float64 `json:"metrics_age_seconds,omitempty"`
}

func margin(total int64, percent, minGiB float64) int64 {
	m := int64(float64(total) * percent / 100)
	if floor := int64(minGiB * gib); floor > m {
		m = floor
	}
	return m
}

func ptr[T any](v T) *T { return &v }

// ComputeResources builds the partition-planning view of a worker.
func ComputeResources(rec registry.WorkerRecord, cfg Config, now time.Time) Resources {
	accelerators := usableAccelerators(rec)
	res := Resources{
		WorkerID:               rec.ID,
		Online:                 rec.Online,
		ProfileSource:          string(rec.ProfileSource),
		RAMTotalBytes:          rec.TotalRAMBytes,
		RAMAvailableBytes:      rec.RAMAvailableBytes,
		RAMSafetyMarginBytes:   margin(rec.TotalRAMBytes, cfg.RAMSafetyMarginPercent, cfg.RAMSafetyMarginMinGiB),
		SwapUtilizationPercent: rec.SwapUtilizationPercent,
		CPUPhysicalCores:       rec.CPUPhysicalCores,
		CPULogicalCores:        rec.CPULogicalCores,
		CPUUtilizationPercent:  rec.CPUUtilizationPercent,
		CPUFrequencyMaxMHz:     rec.CPUFrequencyMaxMHz,
		ExecutionBackends:      []string{},
		BackendGFLOPS:          map[string]float64{},
		BackendProbePending:    rec.BackendProbePending,
		GPUs:                   []GPUResources{},
	}
	if res.ProfileSource == "" {
		res.ProfileSource = string(registry.ProfileSourceUnknown)
	}
	if rec.RAMAvailableBytes != nil {
		usable := *rec.RAMAvailableBytes - res.RAMSafetyMarginBytes
		if usable < 0 {
			usable = 0
		}
		res.RAMUsableBytes = &usable
	}
	if cpuUsable(rec) && !rec.BackendProbePending {
		res.ExecutionBackends = append(res.ExecutionBackends, "cpu")
	}
	res.ExecutionBackends = append(res.ExecutionBackends, accelerators...)
	for _, b := range rec.BackendCapabilities {
		if b.TestGFLOPS != nil && b.Usable() {
			res.BackendGFLOPS[b.Name] = *b.TestGFLOPS
			if b.Name == "cpu" {
				res.CPUMeasuredGFLOPS = b.TestGFLOPS
			}
		}
	}

	metricsByID := map[string]registry.GPUMetrics{}
	for _, m := range rec.GPUMetrics {
		metricsByID[m.GPUID] = m
	}
	var totalUsable int64
	anyUsable := false
	for _, g := range rec.GPUs {
		gr := GPUResources{
			GPUID:         g.ID,
			Name:          g.Name,
			Vendor:        g.Vendor,
			MemoryType:    g.MemoryType,
			ComputeUsable: gpuComputeUsable(g, accelerators),
		}
		m := metricsByID[g.ID]
		gr.UtilizationPercent = m.UtilizationPercent
		if g.MemoryType == "shared" || g.Integrated {
			gr.Note = "integrated GPU: uses system RAM, which is already counted under RAM; no separate VRAM"
		} else {
			total := g.DedicatedMemoryBytes
			if total == nil && g.MemoryType == "dedicated" {
				total = g.MemoryTotalBytes
			}
			gr.VRAMTotalBytes = total
			gr.VRAMUsedBytes = m.DedicatedMemoryUsedBytes
			if total != nil {
				gr.VRAMSafetyMarginBytes = ptr(margin(*total, cfg.VRAMSafetyMarginPercent, cfg.VRAMSafetyMarginMinGiB))
				if m.DedicatedMemoryUsedBytes != nil {
					usable := *total - *m.DedicatedMemoryUsedBytes - *gr.VRAMSafetyMarginBytes
					if usable < 0 {
						usable = 0
					}
					gr.VRAMUsableBytes = &usable
				} else {
					gr.Note = "VRAM in use not reported, so usable VRAM is unknown"
				}
			}
			if gr.ComputeUsable && gr.VRAMUsableBytes != nil {
				totalUsable += *gr.VRAMUsableBytes
				anyUsable = true
			}
		}
		if !gr.ComputeUsable && gr.Note == "" {
			gr.Note = "no usable compute backend targets this GPU"
		}
		res.GPUs = append(res.GPUs, gr)
	}
	if anyUsable {
		res.TotalUsableVRAMBytes = &totalUsable
	}
	if !rec.MetricsUpdatedAt.IsZero() {
		res.MetricsAgeSeconds = ptr(now.Sub(rec.MetricsUpdatedAt).Seconds())
	}
	return res
}
