package sleipnir

import (
	"context"
	"sync"
)

// CompactStore tracks stable compaction watermarks per agent, scoped to one run.
// Stored in ctx by the harness; consumed by the Compactor middleware.
type CompactStore interface {
	GetWatermark(agentName string) int // returns 0 if not set
	SetWatermark(agentName string, n int)
}

type compactStoreKey struct{}

// WithCompactStore stores cs in ctx. Called by the harness at the start of every
// runLoop invocation; also usable in tests for unit-testing middleware that reads
// the store.
func WithCompactStore(ctx context.Context, cs CompactStore) context.Context {
	return context.WithValue(ctx, compactStoreKey{}, cs)
}

// CompactStoreFrom returns the CompactStore added by the harness, or (nil, false).
// Used by middleware/compact — not needed by most callers.
func CompactStoreFrom(ctx context.Context) (CompactStore, bool) {
	cs, ok := ctx.Value(compactStoreKey{}).(CompactStore)
	return cs, ok
}

// syncMapCompactStore implements CompactStore using sync.Map.
type syncMapCompactStore struct {
	m sync.Map
}

func (s *syncMapCompactStore) GetWatermark(agentName string) int {
	v, ok := s.m.Load(agentName)
	if !ok {
		return 0
	}
	return v.(int)
}

func (s *syncMapCompactStore) SetWatermark(agentName string, n int) {
	s.m.Store(agentName, n)
}
