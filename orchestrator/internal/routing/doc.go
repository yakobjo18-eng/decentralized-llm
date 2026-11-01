// Package routing will select, for each pipeline stage, which eligible
// registered worker executes it, and handle rerouting when a worker
// becomes unavailable. Not yet implemented -- lands alongside Phase 4
// (initial assignment) and Phase 8 (failure rerouting). Per
// docs/ARCHITECTURE.md, this is explicitly not shortest-path routing:
// Transformer layers are a fixed sequential chain, not a graph.
package routing
