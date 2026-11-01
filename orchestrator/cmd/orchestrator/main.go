// Command orchestrator runs the Go control-plane process: a
// WorkerRegistry gRPC service (for worker registration/heartbeats) and
// a REST API (for the dashboard and operators). See
// docs/ARCHITECTURE.md and docs/RUNNING_PHASE1.md.
package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"time"

	"google.golang.org/grpc"

	"orchestrator/internal/api"
	"orchestrator/internal/config"
	"orchestrator/internal/grpcserver"
	registrypb "orchestrator/internal/proto/registry"
	"orchestrator/internal/registry"
)

func main() {
	configPath := flag.String("config", "orchestrator.yaml", "path to the orchestrator config file (missing file is fine; defaults + env vars still apply)")
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}
	log.Printf("config loaded: rest=%s grpc=%s log_level=%s cors_origin=%q heartbeat_timeout=%.1fs allow_simulated_workers=%t",
		cfg.RESTAddress, cfg.GRPCAddress, cfg.LogLevel, cfg.CORSOrigin, cfg.HeartbeatTimeoutSeconds, cfg.AllowSimulatedWorkers)
	for _, addr := range []string{cfg.RESTAddress, cfg.GRPCAddress} {
		if warning, _ := config.CheckBindAddress(addr); warning != "" {
			log.Printf("WARNING: %s", warning)
		}
	}
	if cfg.AllowSimulatedWorkers {
		log.Printf("WARNING: accepting SIMULATED hardware profiles -- they are %s and are labelled as such in every API response",
			registry.SimulatedNotice)
	}

	heartbeatTimeout := time.Duration(cfg.HeartbeatTimeoutSeconds * float64(time.Second))
	store := registry.NewInMemoryRegistryWithTimeout(heartbeatTimeout)

	// --- gRPC server (WorkerRegistry: Register/Heartbeat/Deregister) ---
	lis, err := net.Listen("tcp", cfg.GRPCAddress)
	if err != nil {
		log.Fatalf("failed to listen on gRPC address %s: %v", cfg.GRPCAddress, err)
	}
	grpcServer := grpc.NewServer()
	registrypb.RegisterWorkerRegistryServer(grpcServer, grpcserver.NewRegistryServerWithOptions(store, grpcserver.Options{
		AllowSimulatedWorkers: cfg.AllowSimulatedWorkers,
	}))

	grpcErrCh := make(chan error, 1)
	go func() {
		log.Printf("gRPC WorkerRegistry service listening on %s", cfg.GRPCAddress)
		if err := grpcServer.Serve(lis); err != nil {
			grpcErrCh <- err
		}
	}()

	// --- REST API (dashboard-facing: /health, /api/workers) ---
	restServer := api.NewServerWithCapability(store, cfg.CORSOrigin, cfg.Capability)
	httpServer := &http.Server{
		Addr:    cfg.RESTAddress,
		Handler: restServer.Handler(),
	}

	httpErrCh := make(chan error, 1)
	go func() {
		log.Printf("REST API listening on %s", cfg.RESTAddress)
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			httpErrCh <- err
		}
	}()

	// --- wait for shutdown signal or a server failure ---
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt)

	select {
	case <-stop:
		log.Println("shutdown signal received")
	case err := <-grpcErrCh:
		log.Printf("gRPC server failed: %v", err)
	case err := <-httpErrCh:
		log.Printf("REST server failed: %v", err)
	}

	log.Println("shutting down...")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := httpServer.Shutdown(ctx); err != nil {
		log.Printf("REST server shutdown error: %v", err)
	}
	grpcServer.GracefulStop()

	log.Println("shutdown complete")
}
