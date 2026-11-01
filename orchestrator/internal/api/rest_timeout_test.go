package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"orchestrator/internal/registry"
)

func fetchWorkers(t *testing.T, srv *Server) []workerView {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/api/workers", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	var workers []workerView
	if err := json.Unmarshal(rec.Body.Bytes(), &workers); err != nil {
		t.Fatalf("invalid JSON body: %v", err)
	}
	return workers
}

func TestHandleWorkers_ReflectsOfflineAfterTimeout(t *testing.T) {
	store := registry.NewInMemoryRegistryWithTimeout(30 * time.Millisecond)
	if _, err := store.Register(registry.WorkerRecord{ID: "w1", Address: "a", Hostname: "h"}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	srv := NewServer(store, "")

	workers := fetchWorkers(t, srv)
	if len(workers) != 1 || !workers[0].Online {
		t.Fatalf("expected one online worker immediately after registering, got %+v", workers)
	}

	time.Sleep(60 * time.Millisecond)

	workers = fetchWorkers(t, srv)
	if len(workers) != 1 {
		t.Fatalf("offline worker should still be listed, got %d workers", len(workers))
	}
	if workers[0].Online {
		t.Error("worker should be reported offline after the heartbeat timeout")
	}
	if workers[0].LastHeartbeatAgeSeconds <= 0 {
		t.Errorf("LastHeartbeatAgeSeconds = %v, want > 0", workers[0].LastHeartbeatAgeSeconds)
	}
}

func TestHandleHealth_CountsOnlineSeparatelyFromRegistered(t *testing.T) {
	store := registry.NewInMemoryRegistryWithTimeout(30 * time.Millisecond)
	if _, err := store.Register(registry.WorkerRecord{ID: "stays-online", Address: "a"}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	if _, err := store.Register(registry.WorkerRecord{ID: "goes-offline", Address: "b"}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	srv := NewServer(store, "")

	time.Sleep(60 * time.Millisecond)
	// Refresh only one of the two workers.
	if err := store.Heartbeat("stays-online", "idle", time.Now()); err != nil {
		t.Fatalf("Heartbeat() error = %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	var body healthResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("invalid JSON body: %v", err)
	}
	if body.RegisteredWorkers != 2 {
		t.Errorf("RegisteredWorkers = %d, want 2", body.RegisteredWorkers)
	}
	if body.OnlineWorkers != 1 {
		t.Errorf("OnlineWorkers = %d, want 1", body.OnlineWorkers)
	}
}
