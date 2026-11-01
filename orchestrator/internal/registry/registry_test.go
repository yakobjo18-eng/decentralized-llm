package registry

import (
	"testing"
	"time"
)

func TestRegister_NewWorker(t *testing.T) {
	r := NewInMemoryRegistry()

	isNew, err := r.Register(WorkerRecord{ID: "worker-1", Address: "127.0.0.1:50051"})
	if err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	if !isNew {
		t.Errorf("Register() isNew = false, want true for a first-time worker")
	}

	rec, ok := r.Get("worker-1")
	if !ok {
		t.Fatalf("Get() after Register() found nothing")
	}
	if rec.Status != "idle" {
		t.Errorf("Status = %q, want default %q", rec.Status, "idle")
	}
	if rec.RegisteredAt.IsZero() || rec.LastHeartbeat.IsZero() {
		t.Errorf("RegisteredAt/LastHeartbeat should be set, got %+v", rec)
	}
}

func TestRegister_EmptyIDRejected(t *testing.T) {
	r := NewInMemoryRegistry()
	if _, err := r.Register(WorkerRecord{Address: "127.0.0.1:50051"}); err == nil {
		t.Fatal("Register() with empty ID should return an error")
	}
}

func TestRegister_SameIDTwiceRefreshesNotDuplicates(t *testing.T) {
	// Phase 2 note: this originally re-registered from a DIFFERENT
	// address to prove "refresh, don't duplicate" -- but that is now
	// exactly the case Phase 2's duplicate-ID conflict detection is
	// supposed to reject (see TestRegister_RejectsConflictingAddressWhileOnline
	// in registry_timeout_test.go for that case). Using the same
	// address here preserves this test's original intent -- a second
	// registration of a known ID must refresh, not duplicate -- in a
	// way that's compatible with the new conflict policy: a fast
	// restart re-registering from its own address is not a conflict.
	r := NewInMemoryRegistry()

	if _, err := r.Register(WorkerRecord{ID: "worker-1", Address: "127.0.0.1:50051"}); err != nil {
		t.Fatalf("first Register() error = %v", err)
	}
	first, _ := r.Get("worker-1")

	isNew, err := r.Register(WorkerRecord{ID: "worker-1", Address: "127.0.0.1:50051"})
	if err != nil {
		t.Fatalf("second Register() error = %v", err)
	}
	if isNew {
		t.Errorf("Register() isNew = true on re-registration, want false")
	}

	second, _ := r.Get("worker-1")
	if second.Address != "127.0.0.1:50051" {
		t.Errorf("Address = %q after re-register, want unchanged address", second.Address)
	}
	if !second.RegisteredAt.Equal(first.RegisteredAt) {
		t.Errorf("RegisteredAt changed on re-register: first=%v second=%v", first.RegisteredAt, second.RegisteredAt)
	}

	if len(r.List()) != 1 {
		t.Errorf("List() length = %d, want 1 (re-register must not duplicate)", len(r.List()))
	}
}

func TestHeartbeat_UpdatesLastSeenAndStatus(t *testing.T) {
	r := NewInMemoryRegistry()
	if _, err := r.Register(WorkerRecord{ID: "worker-1", Address: "127.0.0.1:50051"}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	beforeHeartbeat := time.Now()
	hbTime := beforeHeartbeat.Add(time.Second)
	if err := r.Heartbeat("worker-1", "busy", hbTime); err != nil {
		t.Fatalf("Heartbeat() error = %v", err)
	}

	rec, _ := r.Get("worker-1")
	if rec.Status != "busy" {
		t.Errorf("Status = %q, want busy", rec.Status)
	}
	if !rec.LastHeartbeat.Equal(hbTime) {
		t.Errorf("LastHeartbeat = %v, want %v", rec.LastHeartbeat, hbTime)
	}
}

func TestHeartbeat_UnknownWorkerErrors(t *testing.T) {
	r := NewInMemoryRegistry()
	if err := r.Heartbeat("ghost", "idle", time.Now()); err == nil {
		t.Fatal("Heartbeat() for an unregistered worker should error")
	}
}

func TestDeregister_RemovesWorker(t *testing.T) {
	r := NewInMemoryRegistry()
	if _, err := r.Register(WorkerRecord{ID: "worker-1", Address: "127.0.0.1:50051"}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	if err := r.Deregister("worker-1"); err != nil {
		t.Fatalf("Deregister() error = %v", err)
	}
	if _, ok := r.Get("worker-1"); ok {
		t.Error("worker still present after Deregister()")
	}
}

func TestDeregister_UnknownWorkerErrors(t *testing.T) {
	r := NewInMemoryRegistry()
	if err := r.Deregister("ghost"); err == nil {
		t.Fatal("Deregister() for an unregistered worker should error")
	}
}

func TestList_SortedByID(t *testing.T) {
	r := NewInMemoryRegistry()
	for _, id := range []string{"worker-3", "worker-1", "worker-2"} {
		if _, err := r.Register(WorkerRecord{ID: id, Address: "127.0.0.1:50051"}); err != nil {
			t.Fatalf("Register(%q) error = %v", id, err)
		}
	}

	got := r.List()
	if len(got) != 3 {
		t.Fatalf("List() length = %d, want 3", len(got))
	}
	for i, want := range []string{"worker-1", "worker-2", "worker-3"} {
		if got[i].ID != want {
			t.Errorf("List()[%d].ID = %q, want %q", i, got[i].ID, want)
		}
	}
}
