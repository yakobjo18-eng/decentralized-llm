package grpcserver

import (
	"context"
	"testing"

	commonpb "orchestrator/internal/proto/common"
	registrypb "orchestrator/internal/proto/registry"
	"orchestrator/internal/registry"
)

func TestRegister_MapsHardwareFieldsIntoRegistry(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	srv := NewRegistryServer(store)

	resp, err := srv.Register(context.Background(), &registrypb.RegisterRequest{
		WorkerId: "worker-1",
		Address:  "127.0.0.1:50051",
		Hardware: &commonpb.HardwareInfo{
			Hostname:        "WORKER-PC",
			CpuLogicalCores: 12,
			TotalRamBytes:   16 << 30,
			HasCudaGpu:      false,
			Os:              "windows",
			PythonVersion:   "3.13.3",
		},
	})
	if err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	if !resp.GetAccepted() {
		t.Fatalf("Register() not accepted: %s", resp.GetMessage())
	}

	rec, ok := store.Get("worker-1")
	if !ok {
		t.Fatal("worker not found in registry after Register()")
	}
	if rec.Hostname != "WORKER-PC" || rec.CPULogicalCores != 12 || rec.TotalRAMBytes != 16<<30 {
		t.Errorf("hardware fields not mapped correctly: %+v", rec)
	}
}

func TestRegister_EmptyWorkerIDRejectedWithoutError(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	srv := NewRegistryServer(store)

	resp, err := srv.Register(context.Background(), &registrypb.RegisterRequest{Address: "127.0.0.1:50051"})
	if err != nil {
		t.Fatalf("Register() should not return a transport error for a rejected request, got %v", err)
	}
	if resp.GetAccepted() {
		t.Error("Register() with empty worker_id should not be accepted")
	}
}

func TestHeartbeat_UnknownWorkerNotAcknowledged(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	srv := NewRegistryServer(store)

	resp, err := srv.Heartbeat(context.Background(), &registrypb.HeartbeatRequest{WorkerId: "ghost"})
	if err != nil {
		t.Fatalf("Heartbeat() error = %v", err)
	}
	if resp.GetAcknowledged() {
		t.Error("Heartbeat() for an unregistered worker should not be acknowledged")
	}
}

func TestDeregister_RemovesFromRegistry(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	srv := NewRegistryServer(store)

	if _, err := srv.Register(context.Background(), &registrypb.RegisterRequest{WorkerId: "worker-1", Address: "x"}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	resp, err := srv.Deregister(context.Background(), &registrypb.DeregisterRequest{WorkerId: "worker-1"})
	if err != nil {
		t.Fatalf("Deregister() error = %v", err)
	}
	if !resp.GetAccepted() {
		t.Error("Deregister() of a known worker should be accepted")
	}
	if _, ok := store.Get("worker-1"); ok {
		t.Error("worker still present in registry after Deregister()")
	}
}
