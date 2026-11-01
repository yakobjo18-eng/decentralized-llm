package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"orchestrator/internal/registry"
)

func TestHandleHealth_EmptyRegistry(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	srv := NewServer(store, "")

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	var body healthResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("invalid JSON body: %v", err)
	}
	if body.Status != "ok" {
		t.Errorf("Status = %q, want ok", body.Status)
	}
	if body.RegisteredWorkers != 0 {
		t.Errorf("RegisteredWorkers = %d, want 0 on an empty registry", body.RegisteredWorkers)
	}
}

func TestHandleWorkers_ReflectsRealRegistryState(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	if _, err := store.Register(registry.WorkerRecord{
		ID:              "worker-1",
		Address:         "127.0.0.1:50051",
		Hostname:        "TEST-PC",
		CPULogicalCores: 8,
		TotalRAMBytes:   16 << 30,
	}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	srv := NewServer(store, "")

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
	if len(workers) != 1 {
		t.Fatalf("got %d workers, want 1", len(workers))
	}
	if workers[0].ID != "worker-1" || workers[0].Hostname != "TEST-PC" || workers[0].CPULogicalCores != 8 {
		t.Errorf("worker data did not round-trip correctly: %+v", workers[0])
	}
}

func TestHandleHealth_RejectsNonGET(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	srv := NewServer(store, "")

	req := httptest.NewRequest(http.MethodPost, "/health", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", rec.Code)
	}
}

func TestCORS_SetsConfiguredOrigin(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	srv := NewServer(store, "http://localhost:3000")

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "http://localhost:3000" {
		t.Errorf("Access-Control-Allow-Origin = %q, want http://localhost:3000", got)
	}
}
