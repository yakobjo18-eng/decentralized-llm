package grpcserver

import (
	"context"
	"testing"

	registrypb "orchestrator/internal/proto/registry"
	"orchestrator/internal/registry"
)

func TestRegister_ConflictingReregistrationRejectedWithClearMessage(t *testing.T) {
	store := registry.NewInMemoryRegistry() // default timeout, definitely still online moments after registering
	srv := NewRegistryServer(store)

	if _, err := srv.Register(context.Background(), &registrypb.RegisterRequest{WorkerId: "w1", Address: "10.0.0.1:50100"}); err != nil {
		t.Fatalf("first Register() error = %v", err)
	}

	resp, err := srv.Register(context.Background(), &registrypb.RegisterRequest{WorkerId: "w1", Address: "10.0.0.2:50100"})
	if err != nil {
		t.Fatalf("Register() should not return a transport error for a rejected request, got %v", err)
	}
	if resp.GetAccepted() {
		t.Error("a conflicting registration from a different address while the original is still online should not be accepted")
	}
	if resp.GetMessage() == "" {
		t.Error("a rejected registration should explain why, so the worker's startup logs are useful")
	}
}

func TestRegister_SameAddressReregistrationStillAccepted(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	srv := NewRegistryServer(store)

	if _, err := srv.Register(context.Background(), &registrypb.RegisterRequest{WorkerId: "w1", Address: "10.0.0.1:50100"}); err != nil {
		t.Fatalf("first Register() error = %v", err)
	}
	resp, err := srv.Register(context.Background(), &registrypb.RegisterRequest{WorkerId: "w1", Address: "10.0.0.1:50100"})
	if err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	if !resp.GetAccepted() {
		t.Errorf("a fast restart re-registering from the same address should be accepted, got message: %s", resp.GetMessage())
	}
}
