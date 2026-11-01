package config

import (
	"os"
	"path/filepath"
	"testing"

	"orchestrator/internal/capability"
)

func TestDefault(t *testing.T) {
	cfg := Default()
	if cfg.RESTAddress == "" || cfg.GRPCAddress == "" {
		t.Fatalf("Default() must set non-empty addresses, got %+v", cfg)
	}
}

func TestLoad_MissingFileUsesDefaults(t *testing.T) {
	cfg, err := Load(filepath.Join(t.TempDir(), "does-not-exist.yaml"))
	if err != nil {
		t.Fatalf("Load() with a missing file should not error, got: %v", err)
	}
	if cfg != Default() {
		t.Fatalf("expected defaults, got %+v", cfg)
	}
}

func TestLoad_FileOverridesDefaults(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "orchestrator.yaml")
	contents := "rest_address: \"127.0.0.1:9999\"\ngrpc_address: \"127.0.0.1:9998\"\nlog_level: \"debug\"\n"
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatalf("failed to write test config: %v", err)
	}

	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load() failed: %v", err)
	}
	if cfg.RESTAddress != "127.0.0.1:9999" {
		t.Errorf("RESTAddress = %q, want 127.0.0.1:9999", cfg.RESTAddress)
	}
	if cfg.GRPCAddress != "127.0.0.1:9998" {
		t.Errorf("GRPCAddress = %q, want 127.0.0.1:9998", cfg.GRPCAddress)
	}
	if cfg.LogLevel != "debug" {
		t.Errorf("LogLevel = %q, want debug", cfg.LogLevel)
	}
	// Untouched by the file, should still be the default.
	if cfg.CORSOrigin != Default().CORSOrigin {
		t.Errorf("CORSOrigin = %q, want default %q", cfg.CORSOrigin, Default().CORSOrigin)
	}
}

func TestLoad_EnvOverridesFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "orchestrator.yaml")
	contents := "rest_address: \"127.0.0.1:9999\"\n"
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatalf("failed to write test config: %v", err)
	}

	t.Setenv("ORCHESTRATOR_REST_ADDRESS", "10.0.0.5:7000")

	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load() failed: %v", err)
	}
	if cfg.RESTAddress != "10.0.0.5:7000" {
		t.Errorf("env override did not win: RESTAddress = %q, want 10.0.0.5:7000", cfg.RESTAddress)
	}
}

func TestLoad_HeartbeatTimeoutFromFileAndEnv(t *testing.T) {
	if Default().HeartbeatTimeoutSeconds <= 0 {
		t.Fatalf("Default() HeartbeatTimeoutSeconds = %v, want a positive default", Default().HeartbeatTimeoutSeconds)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "orchestrator.yaml")
	if err := os.WriteFile(path, []byte("heartbeat_timeout_seconds: 42\n"), 0o644); err != nil {
		t.Fatalf("failed to write test config: %v", err)
	}

	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load() failed: %v", err)
	}
	if cfg.HeartbeatTimeoutSeconds != 42 {
		t.Errorf("HeartbeatTimeoutSeconds = %v, want 42 (from file)", cfg.HeartbeatTimeoutSeconds)
	}

	t.Setenv("ORCHESTRATOR_HEARTBEAT_TIMEOUT_SECONDS", "7.5")
	cfg, err = Load(path)
	if err != nil {
		t.Fatalf("Load() failed: %v", err)
	}
	if cfg.HeartbeatTimeoutSeconds != 7.5 {
		t.Errorf("env override did not win: HeartbeatTimeoutSeconds = %v, want 7.5", cfg.HeartbeatTimeoutSeconds)
	}
}

func TestDefault_SimulatedWorkersRejectedAndCapabilityDefaults(t *testing.T) {
	cfg := Default()
	if cfg.AllowSimulatedWorkers {
		t.Error("simulated workers must be opt-in")
	}
	if cfg.Capability != capability.DefaultConfig() {
		t.Errorf("Capability = %+v, want defaults", cfg.Capability)
	}
}

func TestLoad_CapabilitySectionPartialOverride(t *testing.T) {
	path := filepath.Join(t.TempDir(), "orchestrator.yaml")
	contents := "allow_simulated_workers: true\ncapability:\n  high_memory_threshold_gib: 48\n"
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.AllowSimulatedWorkers || cfg.Capability.HighMemoryThresholdGiB != 48 {
		t.Errorf("file values not applied: %+v", cfg)
	}
	if cfg.Capability.LowMemoryThresholdGiB != capability.DefaultConfig().LowMemoryThresholdGiB {
		t.Error("unspecified capability keys must keep their defaults")
	}
}

func TestLoad_InvalidCapabilityConfigRejected(t *testing.T) {
	path := filepath.Join(t.TempDir(), "orchestrator.yaml")
	if err := os.WriteFile(path, []byte("capability:\n  low_memory_threshold_gib: 64\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil {
		t.Error("low >= high threshold must fail to load")
	}
}

func TestLoad_AllowSimulatedEnvOverride(t *testing.T) {
	t.Setenv("ORCHESTRATOR_ALLOW_SIMULATED_WORKERS", "true")
	cfg, err := Load("")
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.AllowSimulatedWorkers {
		t.Error("env override not applied")
	}
}
