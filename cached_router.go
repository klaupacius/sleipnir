package sleipnir

import (
	"context"
	"sync"
)

// CachedRouter wraps a ModelRouter and caches resolved ModelConfig values
// keyed by agent name. The cache persists for the lifetime of the CachedRouter
// instance. Construct a new CachedRouter per Run to get per-Run scoping.
type CachedRouter struct {
	inner ModelRouter
	mu    sync.RWMutex
	cache map[string]ModelConfig
}

func NewCachedRouter(inner ModelRouter) *CachedRouter {
	return &CachedRouter{
		inner: inner,
		cache: make(map[string]ModelConfig),
	}
}

func (r *CachedRouter) Resolve(ctx context.Context, agentName string) (ModelConfig, error) {
	r.mu.RLock()
	if cfg, ok := r.cache[agentName]; ok {
		r.mu.RUnlock()
		return cfg, nil
	}
	r.mu.RUnlock()

	// Hold the write lock while calling inner so concurrent misses for the same
	// agent name don't produce multiple inner.Resolve calls (TOCTOU).
	r.mu.Lock()
	defer r.mu.Unlock()
	if cfg, ok := r.cache[agentName]; ok {
		return cfg, nil
	}
	cfg, err := r.inner.Resolve(ctx, agentName)
	if err != nil {
		return ModelConfig{}, err
	}
	r.cache[agentName] = cfg
	return cfg, nil
}

var _ ModelRouter = (*CachedRouter)(nil)
