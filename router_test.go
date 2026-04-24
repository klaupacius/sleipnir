package sleipnir

import (
	"context"
	"errors"
	"sync"
	"testing"
)

// countingRouter is a test helper that counts how many times Resolve is called.
type countingRouter struct {
	mu    sync.Mutex
	calls int
	cfg   ModelConfig
	err   error
}

func (r *countingRouter) Resolve(_ context.Context, _ string) (ModelConfig, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls++
	return r.cfg, r.err
}

var _ ModelRouter = MapRouter{}

func TestMapRouterOverride(t *testing.T) {
	want := ModelConfig{Model: "claude-opus-4-5"}
	r := MapRouter{
		Default:   ModelConfig{Model: "claude-haiku-4-5"},
		Overrides: map[string]ModelConfig{"summariser": want},
	}

	got, err := r.Resolve(context.Background(), "summariser")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != want {
		t.Errorf("Resolve(%q) = %v, want %v", "summariser", got, want)
	}
}

func TestMapRouterDefault(t *testing.T) {
	want := ModelConfig{Model: "claude-haiku-4-5"}
	r := MapRouter{
		Default:   want,
		Overrides: map[string]ModelConfig{"summariser": {Model: "claude-opus-4-5"}},
	}

	got, err := r.Resolve(context.Background(), "unknown-agent")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != want {
		t.Errorf("Resolve(%q) = %v, want %v", "unknown-agent", got, want)
	}
}

func TestMapRouterEmptyOverrides(t *testing.T) {
	want := ModelConfig{Model: "claude-haiku-4-5"}
	r := MapRouter{
		Default:   want,
		Overrides: nil,
	}

	got, err := r.Resolve(context.Background(), "any-agent")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != want {
		t.Errorf("Resolve(%q) = %v, want %v", "any-agent", got, want)
	}
}

func TestCachedRouterCachesResult(t *testing.T) {
	want := ModelConfig{Model: "claude-sonnet-4-5"}
	inner := &countingRouter{cfg: want}
	cr := NewCachedRouter(inner)

	for i := range 2 {
		got, err := cr.Resolve(context.Background(), "agent-a")
		if err != nil {
			t.Fatalf("call %d: unexpected error: %v", i, err)
		}
		if got != want {
			t.Errorf("call %d: got %v, want %v", i, got, want)
		}
	}

	inner.mu.Lock()
	calls := inner.calls
	inner.mu.Unlock()
	if calls != 1 {
		t.Errorf("inner Resolve called %d times, want 1", calls)
	}
}

func TestCachedRouterDistinctAgents(t *testing.T) {
	want := ModelConfig{Model: "claude-haiku-4-5"}
	inner := &countingRouter{cfg: want}
	cr := NewCachedRouter(inner)

	for _, name := range []string{"agent-a", "agent-b"} {
		if _, err := cr.Resolve(context.Background(), name); err != nil {
			t.Fatalf("Resolve(%q): unexpected error: %v", name, err)
		}
	}

	inner.mu.Lock()
	calls := inner.calls
	inner.mu.Unlock()
	if calls != 2 {
		t.Errorf("inner Resolve called %d times, want 2", calls)
	}
}

func TestCachedRouterInnerError(t *testing.T) {
	sentinel := errors.New("router failure")
	inner := &countingRouter{err: sentinel}
	cr := NewCachedRouter(inner)

	// First call — expect error.
	if _, err := cr.Resolve(context.Background(), "agent-a"); !errors.Is(err, sentinel) {
		t.Fatalf("first call: got err %v, want %v", err, sentinel)
	}

	// Second call — error must not be cached; inner must be called again.
	if _, err := cr.Resolve(context.Background(), "agent-a"); !errors.Is(err, sentinel) {
		t.Fatalf("second call: got err %v, want %v", err, sentinel)
	}

	inner.mu.Lock()
	calls := inner.calls
	inner.mu.Unlock()
	if calls != 2 {
		t.Errorf("inner Resolve called %d times after two error calls, want 2", calls)
	}
}

func TestCachedRouterConcurrentSafe(t *testing.T) {
	want := ModelConfig{Model: "claude-opus-4-5"}
	inner := &countingRouter{cfg: want}
	cr := NewCachedRouter(inner)

	const goroutines = 10
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for range goroutines {
		go func() {
			defer wg.Done()
			got, err := cr.Resolve(context.Background(), "shared-agent")
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}
			if got != want {
				t.Errorf("got %v, want %v", got, want)
			}
		}()
	}
	wg.Wait()

	inner.mu.Lock()
	calls := inner.calls
	inner.mu.Unlock()
	if calls != 1 {
		t.Errorf("inner Resolve called %d times concurrently, want 1", calls)
	}
}
