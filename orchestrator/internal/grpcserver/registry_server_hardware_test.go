package grpcserver

import (
	"context"
	"testing"

	commonpb "orchestrator/internal/proto/common"
	registrypb "orchestrator/internal/proto/registry"
	"orchestrator/internal/registry"
)

func TestRegister_MapsPhase3HardwareFieldsIncludingGPUs(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	srv := NewRegistryServer(store)

	cudaAvailable := false
	resp, err := srv.Register(context.Background(), &registrypb.RegisterRequest{
		WorkerId: "w1",
		Address:  "127.0.0.1:50100",
		Hardware: &commonpb.HardwareInfo{
			Hostname:              "WORKER-PC",
			CpuLogicalCores:       12,
			TotalRamBytes:         16 << 30,
			HasCudaGpu:            false,
			Os:                    "Windows 11",
			PythonVersion:         "3.13.3",
			HardwareSchemaVersion: 2,
			CpuManufacturer:       "Intel",
			CpuModel:              "13th Gen Intel(R) Core(TM) i7-1355U",
			CpuPhysicalCores:      10,
			CpuArchitecture:       "AMD64",
			OsName:                "Windows",
			OsVersion:             "11",
			OsBuild:               "10.0.26200",
			Gpus: []*commonpb.GPUInfo{
				{
					Vendor:           "Intel",
					Name:             "Intel(R) Iris(R) Xe Graphics",
					MemoryTotalBytes: proto64(2147479552),
					MemoryType:       "shared",
					Backends:         []string{},
					DriverVersion:    "32.0.101.6881",
				},
			},
			TorchInstalled:      false,
			CpuBackendAvailable: false,
			CudaAvailable:       &cudaAvailable,
		},
	})
	if err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	if !resp.GetAccepted() {
		t.Fatalf("Register() not accepted: %s", resp.GetMessage())
	}

	rec, ok := store.Get("w1")
	if !ok {
		t.Fatal("worker not found after Register()")
	}

	if rec.CPUPhysicalCores != 10 || rec.CPULogicalCores != 12 {
		t.Errorf("physical=%d logical=%d, want 10/12", rec.CPUPhysicalCores, rec.CPULogicalCores)
	}
	if rec.CPUManufacturer != "Intel" || rec.CPUModel != "13th Gen Intel(R) Core(TM) i7-1355U" {
		t.Errorf("CPU identity not mapped correctly: %+v", rec)
	}
	if len(rec.GPUs) != 1 {
		t.Fatalf("GPUs length = %d, want 1", len(rec.GPUs))
	}
	if rec.GPUs[0].MemoryType != "shared" || rec.GPUs[0].MemoryTotalBytes == nil || *rec.GPUs[0].MemoryTotalBytes != 2147479552 {
		t.Errorf("GPU not mapped correctly: %+v", rec.GPUs[0])
	}
	if rec.CUDAAvailable == nil || *rec.CUDAAvailable != false {
		t.Errorf("CUDAAvailable = %v, want a known false (not nil/unknown) since the proto explicitly set it", rec.CUDAAvailable)
	}
}

func TestRegister_NilHardwareDoesNotPanic(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	srv := NewRegistryServer(store)

	// No Hardware field at all -- must not panic on the tri-state
	// pointer fields (CudaAvailable etc.), which are raw pointer
	// accesses on a potentially-nil submessage.
	resp, err := srv.Register(context.Background(), &registrypb.RegisterRequest{
		WorkerId: "w1",
		Address:  "127.0.0.1:50100",
	})
	if err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	if !resp.GetAccepted() {
		t.Fatalf("Register() not accepted: %s", resp.GetMessage())
	}

	rec, _ := store.Get("w1")
	if rec.CUDAAvailable != nil {
		t.Errorf("CUDAAvailable = %v, want nil when no hardware was sent at all", *rec.CUDAAvailable)
	}
}

func TestHeartbeat_MapsLiveMetrics(t *testing.T) {
	store := registry.NewInMemoryRegistry()
	srv := NewRegistryServer(store)

	if _, err := srv.Register(context.Background(), &registrypb.RegisterRequest{WorkerId: "w1", Address: "a"}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	cpuPct := 33.3
	resp, err := srv.Heartbeat(context.Background(), &registrypb.HeartbeatRequest{
		WorkerId:              "w1",
		Status:                "busy",
		CpuUtilizationPercent: &cpuPct,
	})
	if err != nil {
		t.Fatalf("Heartbeat() error = %v", err)
	}
	if !resp.GetAcknowledged() {
		t.Fatal("Heartbeat() not acknowledged")
	}

	rec, _ := store.Get("w1")
	if rec.CPUUtilizationPercent == nil || *rec.CPUUtilizationPercent != 33.3 {
		t.Errorf("CPUUtilizationPercent = %v, want 33.3", rec.CPUUtilizationPercent)
	}
}

func proto64(v int64) *int64 { return &v }
