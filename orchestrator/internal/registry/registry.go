// Package registry maintains the orchestrator's view of which workers
// exist, their reported hardware, and their liveness.
//
// Phase 1 scope: an in-memory implementation behind the Registry
// interface, per the project decision to defer Redis until a later
// phase. Anything that depends only on this interface (the gRPC
// server, the REST API) does not need to change when a Redis-backed
// implementation is added later.
//
// Phase 2 adds: a configurable heartbeat timeout, with "online" status
// computed on read (not stored/mutated by a background sweep -- see
// isOnline) so List()/Get() always reflect the current time without a
// separate goroutine racing against Register/Heartbeat/Deregister; and
// duplicate-worker-ID conflict detection in Register (see
// ErrWorkerIDConflict) so a second, different machine cannot silently
// steal an in-use worker ID out from under an active worker.
//
// Phase 3 adds: a much richer WorkerRecord (CPU manufacturer/model/
// physical-vs-logical cores, multi-vendor GPU inventory, PyTorch/CUDA/
// DirectML capability detection, live CPU/RAM utilization) and
// HeartbeatWithMetrics for refreshing the cheap live-metric fields on
// every heartbeat without re-running expensive GPU detection that
// often. See docs/HARDWARE_PROFILING.md for the full design and
// docs/HARDWARE_CAPABILITIES.md for the field reference.
//
// Deliberately NOT yet implemented (see docs/DEVELOPMENT_STATUS.md):
//   - Persistence across orchestrator restarts (would come with a
//     Redis-backed implementation).
//   - Actually excluding offline workers from scheduling -- there is
//     no scheduler yet (Phase 4/7). The Online field computed here is
//     what that future scheduler is expected to filter on.
package registry

import (
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"
)

// ErrWorkerIDConflict is returned by Register when a worker ID is
// already registered, is currently considered online (a heartbeat was
// seen within the configured timeout), and the new registration
// attempt is from a different address. This is the "prevent duplicate
// registrations from silently overwriting an existing worker" case.
//
// It is NOT returned for a re-registration from the same address
// (treated as a refresh, e.g. a fast restart), nor for a registration
// of a worker ID that has gone offline (treated as a legitimate
// reconnection, possibly from a new address).
var ErrWorkerIDConflict = errors.New("worker id is already registered and online from a different address")

// ErrWorkerNotFound is returned (wrapped) by Heartbeat/Deregister for
// an unknown worker ID -- e.g. after an orchestrator restart wiped the
// in-memory registry. The gRPC layer turns it into
// HeartbeatResponse.reregister_required so the worker re-registers.
var ErrWorkerNotFound = errors.New("worker is not registered")

// ErrInvalidHardware is returned (wrapped, with every problem listed)
// by Register when the reported hardware profile is internally
// inconsistent or impossible. See ValidateHardware.
var ErrInvalidHardware = errors.New("invalid hardware profile")

// ErrInvalidMetrics is returned (wrapped) by HeartbeatWithMetrics when
// the live metrics are impossible. The heartbeat is rejected as a
// whole: the stored record is left unchanged.
var ErrInvalidMetrics = errors.New("invalid live metrics")

// ProfileSource says where a worker's hardware numbers came from.
type ProfileSource string

const (
	// ProfileSourceReal: measured on the machine the worker runs on.
	ProfileSourceReal ProfileSource = "REAL"
	// ProfileSourceSimulated: a hand-written test profile describing a
	// hypothetical machine. Never a measurement.
	ProfileSourceSimulated ProfileSource = "SIMULATED"
	// ProfileSourceUnknown: the worker didn't say (hardware schema < 3).
	// Deliberately NOT assumed to be REAL.
	ProfileSourceUnknown ProfileSource = "UNKNOWN"
)

// SimulatedNotice is shown next to every simulated profile in API
// output so it can't be mistaken for measured hardware.
const SimulatedNotice = "SIMULATED - NOT PHYSICALLY VERIFIED"

// LatestHardwareSchemaVersion is the newest hardware schema this
// orchestrator understands. Newer workers are still accepted (proto3
// ignores unknown fields); their extra fields are simply not stored.
const LatestHardwareSchemaVersion = 3

// DefaultHeartbeatTimeout is used by NewInMemoryRegistry(). It is
// generous relative to the worker's default 5-second heartbeat
// interval (3 missed heartbeats) so transient network hiccups --
// Objective 3's "handle temporary network disconnections" -- do not
// flap a healthy worker offline and back.
const DefaultHeartbeatTimeout = 15 * time.Second

// GPUInfo describes one detected GPU device, as reported by a worker
// (Phase 3). MemoryTotalBytes is nil when genuinely unknown -- never
// fabricated -- e.g. an integrated GPU whose "shared" memory isn't a
// fixed dedicated figure, or a vendor/backend that didn't expose a
// reliable number.
type GPUInfo struct {
	Vendor           string // "NVIDIA" | "AMD" | "Intel" | "Unknown"
	Name             string
	MemoryTotalBytes *int64
	MemoryType       string // "dedicated" | "shared" | "unknown"
	Backends         []string
	DriverVersion    string

	// --- schema 3 --- (see GPUInfo in shared/proto/common.proto)
	ID                   string
	DedicatedMemoryBytes *int64
	SharedMemoryBytes    *int64
	MemorySource         string
	Integrated           bool
}

// GPUMetrics are dynamic per-GPU measurements, matched to GPUInfo by
// GPUID. nil = not measurable, never zero.
type GPUMetrics struct {
	GPUID                    string
	UtilizationPercent       *float64
	DedicatedMemoryUsedBytes *int64
	SharedMemoryUsedBytes    *int64
}

// BackendCapability is one inference backend as actually probed on
// the worker (schema 3). Available/TestPassed are tri-state (nil =
// not determined).
type BackendCapability struct {
	Name        string
	Available   *bool
	DeviceCount int32
	Version     string
	Devices     []string
	Tested      bool
	TestPassed  *bool
	TestGFLOPS  *float64
	Detail      string
}

// Usable reports whether the worker actually demonstrated (or, for a
// simulated profile, declared) that this backend works: available,
// and not failed if a compute test was run.
func (b BackendCapability) Usable() bool {
	return b.Available != nil && *b.Available && (b.TestPassed == nil || *b.TestPassed)
}

// LiveMetrics are the cheap, frequently-refreshed measurements sent
// with every heartbeat (Phase 3). A nil pointer means "not reported
// this beat" (e.g. sampling failed on the worker, or an older worker
// that doesn't send these at all) and leaves the registry's
// previously stored value for that field untouched -- it is never
// treated as zero.
type LiveMetrics struct {
	CPUUtilizationPercent *float64
	RAMAvailableBytes     *int64
	RAMUsedBytes          *int64
	RAMUtilizationPercent *float64

	// --- schema 3 ---
	CPUFrequencyCurrentMHz *float64
	CPULoadAverage1m       *float64
	SwapTotalBytes         *int64
	SwapUsedBytes          *int64
	SwapUtilizationPercent *float64
	// Empty = GPU metrics not sampled this beat; stored values kept.
	// Non-empty entries are merged per GPUID.
	GPUMetrics []GPUMetrics
	// Worker-clock time of sampling; zero if not reported.
	CollectedAt time.Time
}

// WorkerRecord is the orchestrator's stored view of one worker.
//
// HardwareSchemaVersion: 1 means only the original Phase 1/2 fields
// (Hostname, CPULogicalCores, TotalRAMBytes, HasCUDAGPU, GPUName,
// GPUVRAMBytes, OS, PythonVersion) are meaningfully populated -- e.g.
// a test fixture built before Phase 3, or (hypothetically) an older
// worker. 0 (the zero value) is treated the same as 1 by readers. 2
// means the Phase 3 fields below are populated too.
type WorkerRecord struct {
	ID              string
	Address         string
	Hostname        string
	CPULogicalCores int32
	TotalRAMBytes   int64
	HasCUDAGPU      bool
	GPUName         string
	GPUVRAMBytes    int64
	OS              string
	PythonVersion   string
	Status          string
	RegisteredAt    time.Time
	LastHeartbeat   time.Time

	// --- Phase 3 ---
	HardwareSchemaVersion int32

	CPUManufacturer  string
	CPUModel         string
	CPUPhysicalCores int32
	CPUArchitecture  string

	OSName    string
	OSVersion string
	OSBuild   string

	GPUs []GPUInfo

	TorchInstalled      bool
	TorchVersion        string
	CPUBackendAvailable bool
	// Tri-state: nil means "unknown" (torch not installed on the
	// worker, so this was never tested), not "false".
	CUDAAvailable     *bool
	CUDADeviceCount   *int32
	DirectMLAvailable *bool

	// Live metrics (see LiveMetrics). Populated at registration time
	// and refreshed by every heartbeat that reports them.
	CPUUtilizationPercent *float64
	RAMAvailableBytes     *int64
	RAMUsedBytes          *int64
	RAMUtilizationPercent *float64

	// --- schema 3 ---
	ProfileSource ProfileSource
	// Worker-clock time the profile was collected (zero if unreported).
	ProfileCollectedAt   time.Time
	SimulatedProfileName string

	CPUFrequencyCurrentMHz *float64
	CPUFrequencyMaxMHz     *float64
	CPULoadAverage1m       *float64
	SwapTotalBytes         *int64
	SwapUsedBytes          *int64
	SwapUtilizationPercent *float64

	BackendCapabilities   []BackendCapability
	TorchCUDABuildVersion string
	// BackendProbePending: the worker registered before its (slow)
	// backend probe finished; the torch/backend fields say nothing yet.
	BackendProbePending bool

	GPUMetrics []GPUMetrics

	// Orchestrator-clock bookkeeping, set by the registry itself
	// (values passed into Register are ignored): when the full hardware
	// profile was last received, and when live metrics last changed.
	HardwareUpdatedAt time.Time
	MetricsUpdatedAt  time.Time
	// Worker-clock time of the most recent live metrics sample.
	MetricsCollectedAt time.Time

	// Online is derived, not stored input: Register() ignores whatever
	// value is passed in. It is only meaningful on values returned by
	// Get()/List(), computed fresh against time.Now() and the
	// registry's configured heartbeat timeout at the moment of the call.
	Online bool
}

// Registry is the storage interface every orchestrator component
// depends on. Swap InMemoryRegistry for a Redis-backed implementation
// later without touching callers.
type Registry interface {
	// Register adds a new worker or refreshes an existing one (by ID).
	// Returns true if this worker ID was not previously known.
	Register(rec WorkerRecord) (isNew bool, err error)
	// Heartbeat updates the last-seen time and status of an already
	// registered worker. Returns an error if the worker is not known.
	// Kept exactly as it was in Phase 1/2 for backward compatibility;
	// it is equivalent to HeartbeatWithMetrics with an empty
	// LiveMetrics (no live-metric fields get overwritten).
	Heartbeat(workerID, status string, at time.Time) error
	// HeartbeatWithMetrics is Heartbeat plus Phase 3's cheap live
	// metrics. Any nil field in metrics leaves the registry's
	// previously stored value for that field untouched.
	HeartbeatWithMetrics(workerID, status string, at time.Time, metrics LiveMetrics) error
	// Deregister removes a worker. Returns an error if it was not known.
	Deregister(workerID string) error
	// Get returns a worker's record, if known.
	Get(workerID string) (WorkerRecord, bool)
	// List returns all known workers, sorted by ID for stable output.
	List() []WorkerRecord
}

// InMemoryRegistry is a goroutine-safe, process-local Registry.
type InMemoryRegistry struct {
	mu               sync.RWMutex
	workers          map[string]WorkerRecord
	heartbeatTimeout time.Duration
}

// NewInMemoryRegistry creates an empty registry using
// DefaultHeartbeatTimeout. Existing Phase 1 callers and tests keep
// working unchanged against this signature.
func NewInMemoryRegistry() *InMemoryRegistry {
	return NewInMemoryRegistryWithTimeout(DefaultHeartbeatTimeout)
}

// NewInMemoryRegistryWithTimeout creates an empty registry with an
// explicit heartbeat timeout: a worker is considered Online as long as
// its last heartbeat (or registration) was within this duration of
// time.Now(). Configured by the orchestrator from
// heartbeat_timeout_seconds (orchestrator.yaml) /
// ORCHESTRATOR_HEARTBEAT_TIMEOUT_SECONDS (env).
func NewInMemoryRegistryWithTimeout(heartbeatTimeout time.Duration) *InMemoryRegistry {
	return &InMemoryRegistry{
		workers:          make(map[string]WorkerRecord),
		heartbeatTimeout: heartbeatTimeout,
	}
}

// HeartbeatTimeout returns the configured timeout, mainly for the
// orchestrator's own /health reporting and tests.
func (r *InMemoryRegistry) HeartbeatTimeout() time.Duration {
	return r.heartbeatTimeout
}

func (r *InMemoryRegistry) Register(rec WorkerRecord) (bool, error) {
	if rec.ID == "" {
		return false, fmt.Errorf("worker id must not be empty")
	}
	if err := ValidateHardware(rec); err != nil {
		return false, err
	}
	if rec.ProfileSource == "" {
		rec.ProfileSource = ProfileSourceUnknown
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	now := time.Now()
	existing, isKnown := r.workers[rec.ID]

	if isKnown {
		wasOnline := r.isOnline(existing, now)
		sameAddress := existing.Address == rec.Address
		if wasOnline && !sameAddress {
			// A different machine (or a misconfigured duplicate worker_id)
			// is trying to take over an ID that's actively in use. Reject
			// rather than silently overwrite the existing worker's
			// connection info out from under it.
			return false, fmt.Errorf(
				"%w: existing address %q, attempted address %q",
				ErrWorkerIDConflict, existing.Address, rec.Address,
			)
		}
		// Either the same worker re-registering from the same address
		// (a fast restart), or the previous registration has gone
		// stale/offline -- both are legitimate, treat as a refresh.
		rec.RegisteredAt = existing.RegisteredAt
	} else {
		rec.RegisteredAt = now
	}

	rec.LastHeartbeat = now
	rec.HardwareUpdatedAt = now
	rec.MetricsUpdatedAt = now
	rec.MetricsCollectedAt = rec.ProfileCollectedAt
	if rec.Status == "" {
		rec.Status = "idle"
	}

	r.workers[rec.ID] = rec
	return !isKnown, nil
}

// isOnline reports whether rec's last heartbeat is recent enough,
// relative to now, to still be considered alive. Must be called with
// r.mu held (read or write).
func (r *InMemoryRegistry) isOnline(rec WorkerRecord, now time.Time) bool {
	return now.Sub(rec.LastHeartbeat) <= r.heartbeatTimeout
}

func (r *InMemoryRegistry) Heartbeat(workerID, status string, at time.Time) error {
	return r.HeartbeatWithMetrics(workerID, status, at, LiveMetrics{})
}

func (r *InMemoryRegistry) HeartbeatWithMetrics(workerID, status string, at time.Time, metrics LiveMetrics) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	rec, ok := r.workers[workerID]
	if !ok {
		return fmt.Errorf("%w: %q", ErrWorkerNotFound, workerID)
	}
	if err := ValidateLiveMetrics(rec, metrics); err != nil {
		return err
	}
	rec.LastHeartbeat = at
	if status != "" {
		rec.Status = status
	}
	changed := false
	setF := func(dst **float64, src *float64) {
		if src != nil {
			*dst = src
			changed = true
		}
	}
	setI := func(dst **int64, src *int64) {
		if src != nil {
			*dst = src
			changed = true
		}
	}
	setF(&rec.CPUUtilizationPercent, metrics.CPUUtilizationPercent)
	setI(&rec.RAMAvailableBytes, metrics.RAMAvailableBytes)
	setI(&rec.RAMUsedBytes, metrics.RAMUsedBytes)
	setF(&rec.RAMUtilizationPercent, metrics.RAMUtilizationPercent)
	setF(&rec.CPUFrequencyCurrentMHz, metrics.CPUFrequencyCurrentMHz)
	setF(&rec.CPULoadAverage1m, metrics.CPULoadAverage1m)
	setI(&rec.SwapTotalBytes, metrics.SwapTotalBytes)
	setI(&rec.SwapUsedBytes, metrics.SwapUsedBytes)
	setF(&rec.SwapUtilizationPercent, metrics.SwapUtilizationPercent)
	if len(metrics.GPUMetrics) > 0 {
		rec.GPUMetrics = mergeGPUMetrics(rec.GPUMetrics, metrics.GPUMetrics)
		changed = true
	}
	if changed {
		rec.MetricsUpdatedAt = at
		if !metrics.CollectedAt.IsZero() {
			rec.MetricsCollectedAt = metrics.CollectedAt
		}
	}
	r.workers[workerID] = rec
	return nil
}

// mergeGPUMetrics returns a new slice (never aliasing the stored one,
// so a record handed out by Get/List can't change underneath its
// reader) with incoming entries replacing stored entries of the same
// GPUID field by field -- a nil incoming field keeps the stored value.
func mergeGPUMetrics(stored, incoming []GPUMetrics) []GPUMetrics {
	byID := make(map[string]GPUMetrics, len(stored)+len(incoming))
	order := make([]string, 0, len(stored)+len(incoming))
	for _, m := range stored {
		if _, seen := byID[m.GPUID]; !seen {
			order = append(order, m.GPUID)
		}
		byID[m.GPUID] = m
	}
	for _, m := range incoming {
		cur, seen := byID[m.GPUID]
		if !seen {
			order = append(order, m.GPUID)
			cur = GPUMetrics{GPUID: m.GPUID}
		}
		if m.UtilizationPercent != nil {
			cur.UtilizationPercent = m.UtilizationPercent
		}
		if m.DedicatedMemoryUsedBytes != nil {
			cur.DedicatedMemoryUsedBytes = m.DedicatedMemoryUsedBytes
		}
		if m.SharedMemoryUsedBytes != nil {
			cur.SharedMemoryUsedBytes = m.SharedMemoryUsedBytes
		}
		byID[m.GPUID] = cur
	}
	out := make([]GPUMetrics, 0, len(order))
	for _, id := range order {
		out = append(out, byID[id])
	}
	return out
}

func (r *InMemoryRegistry) Deregister(workerID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, ok := r.workers[workerID]; !ok {
		return fmt.Errorf("%w: %q", ErrWorkerNotFound, workerID)
	}
	delete(r.workers, workerID)
	return nil
}

func (r *InMemoryRegistry) Get(workerID string) (WorkerRecord, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	rec, ok := r.workers[workerID]
	if !ok {
		return WorkerRecord{}, false
	}
	rec.Online = r.isOnline(rec, time.Now())
	return rec, true
}

// List returns all known workers, sorted by ID for stable output.
// Workers past the heartbeat timeout are still included -- with
// Online=false -- rather than removed; staleness is a status, not a
// deletion. Only Deregister() (or the worker ID being reused by a
// legitimate reconnection) removes an entry.
func (r *InMemoryRegistry) List() []WorkerRecord {
	r.mu.RLock()
	defer r.mu.RUnlock()

	now := time.Now()
	out := make([]WorkerRecord, 0, len(r.workers))
	for _, rec := range r.workers {
		rec.Online = r.isOnline(rec, now)
		out = append(out, rec)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}
