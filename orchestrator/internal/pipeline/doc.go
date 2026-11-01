// Package pipeline will coordinate an end-to-end distributed inference
// request across its assigned workers: submitting the request,
// tracking per-stage progress, and collecting the final output. Not
// yet implemented -- this is Phase 6 work (real distributed inference),
// building on the tensor-transport layer added in Phase 5.
package pipeline
