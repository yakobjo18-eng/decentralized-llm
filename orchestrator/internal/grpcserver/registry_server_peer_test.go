package grpcserver

import (
	"context"
	"net"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	commonpb "orchestrator/internal/proto/common"
	registrypb "orchestrator/internal/proto/registry"
	"orchestrator/internal/registry"
)

// Phase 8: over a real gRPC connection, Register records the peer IP the
// call came from (here loopback), independently of the reported address.
func TestRegister_RecordsObservedPeerIPOverRealGRPC(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	gs := grpc.NewServer()
	registrypb.RegisterWorkerRegistryServer(gs, NewRegistryServer(store))
	go gs.Serve(lis)
	defer gs.Stop()

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	resp, err := registrypb.NewWorkerRegistryClient(conn).Register(context.Background(), &registrypb.RegisterRequest{
		WorkerId: "w-peer",
		Address:  "203.0.113.9:50100", // what the worker claims; deliberately not where it connects from
		Hardware: &commonpb.HardwareInfo{Hostname: "PEER-TEST"},
	})
	if err != nil || !resp.GetAccepted() {
		t.Fatalf("Register() = %v, %v", resp, err)
	}
	rec, ok := store.Get("w-peer")
	if !ok {
		t.Fatal("worker not stored")
	}
	if rec.ObservedPeerIP != "127.0.0.1" {
		t.Errorf("ObservedPeerIP = %q, want 127.0.0.1", rec.ObservedPeerIP)
	}
	if rec.Address != "203.0.113.9:50100" {
		t.Errorf("reported Address must be kept as reported, got %q", rec.Address)
	}
}

func TestRegister_WithoutPeerInfoLeavesObservedPeerEmpty(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	resp, err := NewRegistryServer(store).Register(context.Background(), &registrypb.RegisterRequest{WorkerId: "w"})
	if err != nil || !resp.GetAccepted() {
		t.Fatalf("Register() = %v, %v", resp, err)
	}
	if rec, _ := store.Get("w"); rec.ObservedPeerIP != "" {
		t.Errorf("ObservedPeerIP = %q, want empty", rec.ObservedPeerIP)
	}
}
