package registry

import (
	"errors"
	"strings"
	"testing"
	"time"
)

// Phase 8: two machines can report the same self-described address (e.g.
// both bind their Health server to 127.0.0.1:50100), so the observed peer
// IP also decides whether a re-registration is the same worker.

func TestRegister_SameReportedAddressFromAnotherMachineIsAConflict(t *testing.T) {
	r := NewInMemoryRegistryWithTimeout(time.Hour)
	if _, err := r.Register(WorkerRecord{ID: "w1", Address: "127.0.0.1:50100", ObservedPeerIP: "192.168.1.10"}); err != nil {
		t.Fatalf("first Register() error = %v", err)
	}
	_, err := r.Register(WorkerRecord{ID: "w1", Address: "127.0.0.1:50100", ObservedPeerIP: "192.168.1.21"})
	if !errors.Is(err, ErrWorkerIDConflict) {
		t.Fatalf("error = %v, want ErrWorkerIDConflict", err)
	}
	if !strings.Contains(err.Error(), "peer 192.168.1.10") || !strings.Contains(err.Error(), "peer 192.168.1.21") {
		t.Errorf("conflict message should name both peers, got %q", err)
	}
	rec, _ := r.Get("w1")
	if rec.ObservedPeerIP != "192.168.1.10" {
		t.Errorf("original registration must be unchanged, got peer %q", rec.ObservedPeerIP)
	}
}

func TestRegister_SameAddressSamePeerIsARefresh(t *testing.T) {
	r := NewInMemoryRegistryWithTimeout(time.Hour)
	for i := 0; i < 2; i++ {
		if _, err := r.Register(WorkerRecord{ID: "w1", Address: "127.0.0.1:50100", ObservedPeerIP: "192.168.1.21"}); err != nil {
			t.Fatalf("Register() #%d error = %v", i+1, err)
		}
	}
}

func TestRegister_UnknownPeerKeepsAddressOnlyBehaviour(t *testing.T) {
	r := NewInMemoryRegistryWithTimeout(time.Hour)
	if _, err := r.Register(WorkerRecord{ID: "w1", Address: "10.0.0.1:50100"}); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Register(WorkerRecord{ID: "w1", Address: "10.0.0.1:50100", ObservedPeerIP: "10.0.0.1"}); err != nil {
		t.Errorf("an unknown previous peer must not cause a conflict, got %v", err)
	}
}

func TestRegister_OfflineWorkerMayReturnFromANewPeer(t *testing.T) {
	r := NewInMemoryRegistryWithTimeout(20 * time.Millisecond)
	if _, err := r.Register(WorkerRecord{ID: "w1", Address: "127.0.0.1:50100", ObservedPeerIP: "192.168.1.21"}); err != nil {
		t.Fatal(err)
	}
	time.Sleep(40 * time.Millisecond) // now offline
	if _, err := r.Register(WorkerRecord{ID: "w1", Address: "127.0.0.1:50100", ObservedPeerIP: "192.168.1.35"}); err != nil {
		t.Errorf("a stale registration may be taken over (e.g. after a DHCP change), got %v", err)
	}
	if rec, _ := r.Get("w1"); rec.ObservedPeerIP != "192.168.1.35" {
		t.Errorf("peer = %q, want the new one", rec.ObservedPeerIP)
	}
}
