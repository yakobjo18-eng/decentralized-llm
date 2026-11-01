// Package config loads orchestrator configuration from a YAML file with
// environment-variable overrides, per Rule 11 (all network addresses and
// ports must be configurable, nothing hardcoded).
package config

import (
	"fmt"
	"os"
	"strconv"

	"gopkg.in/yaml.v3"

	"orchestrator/internal/capability"
)

// Config holds the orchestrator's runtime settings.
type Config struct {
	// RESTAddress is where the dashboard-facing HTTP API listens, e.g. "0.0.0.0:8080".
	RESTAddress string `yaml:"rest_address"`
	// GRPCAddress is where the WorkerRegistry gRPC service listens, e.g. "0.0.0.0:9090".
	GRPCAddress string `yaml:"grpc_address"`
	// LogLevel controls log verbosity ("debug", "info", "warn", "error").
	LogLevel string `yaml:"log_level"`
	// CORSOrigin is the single origin allowed to call the REST API from a
	// browser (the frontend dev server). Empty disables CORS headers.
	CORSOrigin string `yaml:"cors_origin"`
	// HeartbeatTimeoutSeconds is how long a worker can go without a
	// heartbeat before the registry reports it as offline (Phase 2).
	// Kept generous relative to the worker's default 5-second heartbeat
	// interval so a couple of missed beats from transient network
	// hiccups don't flap a healthy worker offline.
	HeartbeatTimeoutSeconds float64 `yaml:"heartbeat_timeout_seconds"`
	// AllowSimulatedWorkers (Phase 3) accepts worker registrations whose
	// hardware profile is SIMULATED (a hand-written test profile, see
	// docs/SIMULATED_HARDWARE.md). Off by default: a real deployment
	// should only ever see measured hardware.
	AllowSimulatedWorkers bool `yaml:"allow_simulated_workers"`
	// Capability holds the classification thresholds and partition
	// safety margins (Phase 3, internal/capability).
	Capability capability.Config `yaml:"capability"`
}

// Default returns the configuration used when no file is present and no
// environment overrides are set. Safe for local, single-machine use.
func Default() Config {
	return Config{
		RESTAddress:             "0.0.0.0:8080",
		GRPCAddress:             "0.0.0.0:9090",
		LogLevel:                "info",
		CORSOrigin:              "http://localhost:3000",
		HeartbeatTimeoutSeconds: 15,
		AllowSimulatedWorkers:   false,
		Capability:              capability.DefaultConfig(),
	}
}

// Load reads configuration from the YAML file at path (if it exists),
// starting from Default(), then applies environment-variable overrides
// (which always win over the file). A missing file is not an error --
// Default() plus env overrides is a valid configuration.
func Load(path string) (Config, error) {
	cfg := Default()

	if path != "" {
		data, err := os.ReadFile(path)
		if err != nil {
			if !os.IsNotExist(err) {
				return Config{}, fmt.Errorf("reading config file %q: %w", path, err)
			}
		} else {
			if err := yaml.Unmarshal(data, &cfg); err != nil {
				return Config{}, fmt.Errorf("parsing config file %q: %w", path, err)
			}
		}
	}

	applyEnvOverrides(&cfg)
	if err := cfg.Capability.Validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func applyEnvOverrides(cfg *Config) {
	if v := os.Getenv("ORCHESTRATOR_REST_ADDRESS"); v != "" {
		cfg.RESTAddress = v
	}
	if v := os.Getenv("ORCHESTRATOR_GRPC_ADDRESS"); v != "" {
		cfg.GRPCAddress = v
	}
	if v := os.Getenv("ORCHESTRATOR_LOG_LEVEL"); v != "" {
		cfg.LogLevel = v
	}
	if v := os.Getenv("ORCHESTRATOR_CORS_ORIGIN"); v != "" {
		cfg.CORSOrigin = v
	}
	if v := os.Getenv("ORCHESTRATOR_HEARTBEAT_TIMEOUT_SECONDS"); v != "" {
		if parsed, err := strconv.ParseFloat(v, 64); err == nil {
			cfg.HeartbeatTimeoutSeconds = parsed
		}
	}
	if v := os.Getenv("ORCHESTRATOR_ALLOW_SIMULATED_WORKERS"); v != "" {
		if parsed, err := strconv.ParseBool(v); err == nil {
			cfg.AllowSimulatedWorkers = parsed
		}
	}
}
