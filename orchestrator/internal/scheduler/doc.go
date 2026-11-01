// Package scheduler will implement hardware- and latency-aware worker
// selection for model partitions: a dynamic-programming solution to the
// chain-partitioning / minimize-the-bottleneck-stage problem described
// in docs/ARCHITECTURE.md section 5. Not yet implemented -- this is
// Phase 7 work. Phase 4 starts with a simpler greedy sequential-fill
// partitioner living in model-partitioner/partitioning, not here.
package scheduler
