// Package monitoring will track network latency/jitter/throughput
// between nodes and aggregate worker utilization history for the
// dashboard's Network Monitoring and Logs pages. Not yet implemented.
// Phase 1 exposes only the minimal, real registry snapshot via
// internal/api (GET /health, GET /api/workers); latency measurement is
// Phase 7, structured log aggregation is Phase 9.
package monitoring
