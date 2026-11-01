package registry

import (
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

func mustOnline(t *testing.T, r *InMemoryRegistry, id string) bool {
	t.Helper()
	rec, ok := r.Get(id)
	if !ok {
		t.Fatalf("Get(%q) found nothing", id)
	}
	return rec.Online
}

func TestList_MarksWorkerOfflineAfterHeartbeatTimeout(t *testing.T) {
	r := NewInMemoryRegistryWithTimeout(30 * time.Millisecond)
	if _, err := r.Register(WorkerRecord{ID: "w1", Address: "a"}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	if !mustOnline(t, r, "w1") {
		t.Fatal("worker should be online immediately after registering")
	}

	time.Sleep(60 * time.Millisecond)

	if mustOnline(t, r, "w1") {
		t.Error("worker should be marked offline after the heartbeat timeout elapses")
	}

	found := false
	for _, rec := range r.List() {
		if rec.ID == "w1" {
			found = true
			if rec.Online {
				t.Error("List() should also report the worker offline, not just Get()")
			}
		}
	}
	if !found {
		t.Error("offline worker should still appear in List() -- staleness is a status, not a removal")
	}
}

func TestHeartbeat_BringsWorkerBackOnlineBeforeTimeout(t *testing.T) {
	r := NewInMemoryRegistryWithTimeout(50 * time.Millisecond)
	if _, err := r.Register(WorkerRecord{ID: "w1", Address: "a"}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	time.Sleep(30 * time.Millisecond)
	if err := r.Heartbeat("w1", "idle", time.Now()); err != nil {
		t.Fatalf("Heartbeat() error = %v", err)
	}
	time.Sleep(30 * time.Millisecond) // 60ms since register, but only 30ms since the heartbeat above

	if !mustOnline(t, r, "w1") {
		t.Error("a recent heartbeat should keep the worker online even though total time since registration exceeds the timeout")
	}
}

func TestRegister_RejectsConflictingAddressWhileOnline(t *testing.T) {
	r := NewInMemoryRegistryWithTimeout(time.Hour) // long timeout: definitely still "online"
	if _, err := r.Register(WorkerRecord{ID: "w1", Address: "10.0.0.1:50100"}); err != nil {
		t.Fatalf("first Register() error = %v", err)
	}

	_, err := r.Register(WorkerRecord{ID: "w1", Address: "10.0.0.2:50100"})
	if err == nil {
		t.Fatal("Register() with a different address for an already-online worker ID should be rejected")
	}
	if !errors.Is(err, ErrWorkerIDConflict) {
		t.Errorf("error = %v, want wrapping ErrWorkerIDConflict", err)
	}

	rec, _ := r.Get("w1")
	if rec.Address != "10.0.0.1:50100" {
		t.Errorf("original registration should be unchanged after a rejected conflicting registration, got address %q", rec.Address)
	}
}

func TestRegister_SameAddressReRegisterIsNotAConflict(t *testing.T) {
	r := NewInMemoryRegistryWithTimeout(time.Hour)
	if _, err := r.Register(WorkerRecord{ID: "w1", Address: "10.0.0.1:50100"}); err != nil {
		t.Fatalf("first Register() error = %v", err)
	}

	isNew, err := r.Register(WorkerRecord{ID: "w1", Address: "10.0.0.1:50100"})
	if err != nil {
		t.Fatalf("re-registration from the same address should be allowed, got error: %v", err)
	}
	if isNew {
		t.Error("re-registration should not report isNew = true")
	}
}

func TestRegister_AllowsReconnectionAfterGoingOffline(t *testing.T) {
	r := NewInMemoryRegistryWithTimeout(30 * time.Millisecond)
	if _, err := r.Register(WorkerRecord{ID: "w1", Address: "10.0.0.1:50100"}); err != nil {
		t.Fatalf("first Register() error = %v", err)
	}

	time.Sleep(60 * time.Millisecond) // now offline

	// A different address is allowed once the previous registration is
	// stale -- this is the "restart on a different port" / genuine
	// reconnect case.
	isNew, err := r.Register(WorkerRecord{ID: "w1", Address: "10.0.0.2:50200"})
	if err != nil {
		t.Fatalf("reconnection after going offline should be allowed, got error: %v", err)
	}
	if isNew {
		t.Error("reconnection of a known (if offline) worker ID should not report isNew = true")
	}

	rec, _ := r.Get("w1")
	if rec.Address != "10.0.0.2:50200" {
		t.Errorf("address should be updated on reconnection, got %q", rec.Address)
	}
	if !rec.Online {
		t.Error("worker should be online again immediately after a successful reconnection")
	}
}

func TestRegister_ConcurrentRegistrationsOfDistinctIDsAreAllSaved(t *testing.T) {
	r := NewInMemoryRegistry()
	const n = 50

	var wg sync.WaitGroup
	errs := make([]error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			id := fmt.Sprintf("worker-%02d", i)
			_, errs[i] = r.Register(WorkerRecord{ID: id, Address: id})
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("Register(worker-%02d) error = %v", i, err)
		}
	}
	if got := len(r.List()); got != n {
		t.Errorf("List() length = %d, want %d (no registrations should be lost to a race)", got, n)
	}
}

func TestRegister_ConcurrentConflictingRegistrationsExactlyOneWins(t *testing.T) {
	r := NewInMemoryRegistryWithTimeout(time.Hour)

	var wg sync.WaitGroup
	results := make([]error, 2)
	addresses := []string{"10.0.0.1:50100", "10.0.0.2:50100"}
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, results[i] = r.Register(WorkerRecord{ID: "contested", Address: addresses[i]})
		}(i)
	}
	wg.Wait()

	accepted := 0
	for _, err := range results {
		if err == nil {
			accepted++
		}
	}
	if accepted != 1 {
		t.Errorf("exactly one of two concurrent conflicting registrations should be accepted, got %d", accepted)
	}
	if len(r.List()) != 1 {
		t.Errorf("registry should contain exactly one worker, got %d", len(r.List()))
	}
}

func TestConcurrentHeartbeatsAndListDoNotRace(t *testing.T) {
	r := NewInMemoryRegistry()
	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("worker-%02d", i)
		if _, err := r.Register(WorkerRecord{ID: id, Address: id}); err != nil {
			t.Fatalf("Register(%q) error = %v", id, err)
		}
	}

	var wg sync.WaitGroup
	stop := make(chan struct{})

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			id := fmt.Sprintf("worker-%02d", i)
			for {
				select {
				case <-stop:
					return
				default:
					_ = r.Heartbeat(id, "busy", time.Now())
				}
			}
		}(i)
	}
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					_ = r.List()
				}
			}
		}()
	}

	time.Sleep(50 * time.Millisecond)
	close(stop)
	wg.Wait()
	// Reaching here (especially under `go test -race`) without a panic
	// or a detected race is the assertion.
}
