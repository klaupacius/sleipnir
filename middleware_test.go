package sleipnir_test

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	sleipnir "sleipnir.dev/sleipnir"
	"sleipnir.dev/sleipnir/sleipnirtest"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

// ---- inline middleware helpers ----

// captureRewriter appends a user message to every LLMRequest before it is sent.
type captureRewriter struct {
	sleipnir.BaseMiddleware
	extra string
}

func (r *captureRewriter) RewriteBeforeLLMCall(_ context.Context, req *sleipnir.LLMRequest) error {
	req.Messages = append(req.Messages, anyllm.Message{Role: anyllm.RoleUser, Content: r.extra})
	return nil
}

// errorRewriter always returns an error.
type errorRewriter struct {
	sleipnir.BaseMiddleware
	err error
}

func (r *errorRewriter) RewriteBeforeLLMCall(_ context.Context, _ *sleipnir.LLMRequest) error {
	return r.err
}

// captureLLMObserver records the last req/resp/err passed to OnLLMCall.
type captureLLMObserver struct {
	sleipnir.BaseMiddleware
	mu   sync.Mutex
	req  *sleipnir.LLMRequest
	resp *sleipnir.LLMResponse
	err  error
	seen bool
}

func (o *captureLLMObserver) OnLLMCall(_ context.Context, req *sleipnir.LLMRequest, resp *sleipnir.LLMResponse, err error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.req = req
	o.resp = resp
	o.err = err
	o.seen = true
}

func (o *captureLLMObserver) wasSeen() bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.seen
}

// captureToolObserver records ToolCall.ToolName for every tool dispatch.
type captureToolObserver struct {
	sleipnir.BaseMiddleware
	mu        sync.Mutex
	toolNames []string
}

func (o *captureToolObserver) OnToolCall(_ context.Context, call *sleipnir.ToolCall, _ *sleipnir.ToolResult, _ error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.toolNames = append(o.toolNames, call.ToolName)
}

func (o *captureToolObserver) names() []string {
	o.mu.Lock()
	defer o.mu.Unlock()
	cp := make([]string, len(o.toolNames))
	copy(cp, o.toolNames)
	return cp
}

// alwaysRetryPolicy always says retry with zero backoff.
type alwaysRetryPolicy struct {
	sleipnir.BaseMiddleware
}

func (p *alwaysRetryPolicy) ShouldRetry(_ context.Context, _ int, _ error) (bool, time.Duration) {
	return true, 0
}

// neverRetryPolicy always says do not retry.
type neverRetryPolicy struct {
	sleipnir.BaseMiddleware
}

func (p *neverRetryPolicy) ShouldRetry(_ context.Context, _ int, _ error) (bool, time.Duration) {
	return false, 0
}

// errorProvider returns an error on every call.
type errorProvider struct {
	err     error
	calls   atomic.Int64
}

func (p *errorProvider) Name() string { return "error_provider" }

func (p *errorProvider) Completion(_ context.Context, _ anyllm.CompletionParams) (*anyllm.ChatCompletion, error) {
	p.calls.Add(1)
	return nil, p.err
}

func (p *errorProvider) CompletionStream(_ context.Context, _ anyllm.CompletionParams) (<-chan anyllm.ChatCompletionChunk, <-chan error) {
	panic("errorProvider: CompletionStream not implemented")
}

// failThenSucceedProvider fails for the first N calls, then returns a text response.
type failThenSucceedProvider struct {
	failFor int
	err     error
	calls   atomic.Int64
}

func (p *failThenSucceedProvider) Name() string { return "fail_then_succeed" }

func (p *failThenSucceedProvider) Completion(_ context.Context, _ anyllm.CompletionParams) (*anyllm.ChatCompletion, error) {
	n := int(p.calls.Add(1))
	if n <= p.failFor {
		return nil, p.err
	}
	return sleipnirtest.TextResponse("ok"), nil
}

func (p *failThenSucceedProvider) CompletionStream(_ context.Context, _ anyllm.CompletionParams) (<-chan anyllm.ChatCompletionChunk, <-chan error) {
	panic("failThenSucceedProvider: CompletionStream not implemented")
}

// captureProvider records CompletionParams for each call, then delegates to inner.
type captureProvider struct {
	inner  anyllm.Provider
	mu     sync.Mutex
	params []anyllm.CompletionParams
}

func (p *captureProvider) Name() string { return "capture_provider" }

func (p *captureProvider) Completion(ctx context.Context, params anyllm.CompletionParams) (*anyllm.ChatCompletion, error) {
	// Deep-copy messages to avoid aliasing with the history slice in runLoop.
	cp := params
	cp.Messages = make([]anyllm.Message, len(params.Messages))
	copy(cp.Messages, params.Messages)
	p.mu.Lock()
	p.params = append(p.params, cp)
	p.mu.Unlock()
	return p.inner.Completion(ctx, params)
}

func (p *captureProvider) CompletionStream(_ context.Context, _ anyllm.CompletionParams) (<-chan anyllm.ChatCompletionChunk, <-chan error) {
	panic("captureProvider: CompletionStream not implemented")
}

func (p *captureProvider) allParams() []anyllm.CompletionParams {
	p.mu.Lock()
	defer p.mu.Unlock()
	cp := make([]anyllm.CompletionParams, len(p.params))
	copy(cp, p.params)
	return cp
}

// ---- tests ----

// TestContextRewriterMutatesRequest: rewriter appends an extra user message;
// the extra message must appear in the params received by the provider.
func TestContextRewriterMutatesRequest(t *testing.T) {
	inner := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("done"))
	cap := &captureProvider{inner: inner}

	rw := &captureRewriter{extra: "injected-message"}
	h := mustNewHarness(t, sleipnir.Config{
		Middlewares: []sleipnir.Middleware{rw},
	})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "a", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}
	_, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "a",
		Prompt:    "hello",
		Router:    defaultRouter(cap),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	allParams := cap.allParams()
	if len(allParams) == 0 {
		t.Fatal("provider was never called")
	}
	msgs := allParams[0].Messages
	found := false
	for _, m := range msgs {
		if m.Content == "injected-message" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected injected-message in provider params, got %v", msgs)
	}
}

// TestContextRewriterError: rewriter returns an error → ErrorEvent is emitted; run still completes.
func TestContextRewriterError(t *testing.T) {
	stub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("done"))
	rw := &errorRewriter{err: errors.New("rewrite error")}

	col := sleipnirtest.NewEventCollector()
	h := mustNewHarness(t, sleipnir.Config{
		Middlewares: []sleipnir.Middleware{rw},
	})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "a", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}
	out, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "a",
		Prompt:    "hi",
		Router:    defaultRouter(stub),
		Events:    col,
	})
	if err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if out.Stopped != sleipnir.StopDone {
		t.Errorf("expected StopDone, got %v", out.Stopped)
	}
	// ErrorEvent should have been emitted for the rewriter error.
	found := false
	for _, e := range col.Events() {
		if ee, ok := e.(sleipnir.ErrorEvent); ok {
			if ee.Err != nil && ee.Err.Error() == "rewrite error" {
				found = true
				break
			}
		}
	}
	if !found {
		t.Error("expected ErrorEvent for rewrite error, none found")
	}
}

// TestLLMObserverCalledOnSuccess: observer records req/resp; assert both non-nil after run.
func TestLLMObserverCalledOnSuccess(t *testing.T) {
	stub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("hello"))
	obs := &captureLLMObserver{}

	h := mustNewHarness(t, sleipnir.Config{
		Middlewares: []sleipnir.Middleware{obs},
	})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "a", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}
	_, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "a",
		Prompt:    "hi",
		Router:    defaultRouter(stub),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !obs.wasSeen() {
		t.Fatal("LLMObserver was not called")
	}
	obs.mu.Lock()
	defer obs.mu.Unlock()
	if obs.req == nil {
		t.Error("req is nil")
	}
	if obs.resp == nil {
		t.Error("resp is nil")
	}
	if obs.err != nil {
		t.Errorf("unexpected observer error: %v", obs.err)
	}
}

// TestLLMObserverCalledOnError: provider always returns error; observer is called with the error.
func TestLLMObserverCalledOnError(t *testing.T) {
	provErr := errors.New("provider error")
	prov := &errorProvider{err: provErr}
	obs := &captureLLMObserver{}

	// MaxLLMRetries: 1 is non-zero so resolveDefaults won't override it.
	// With no RetryPolicy in mws, the retry loop breaks immediately after 1 attempt.
	h := mustNewHarness(t, sleipnir.Config{
		MaxLLMRetries: 1,
		Middlewares:   []sleipnir.Middleware{obs},
	})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "a", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}
	_, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "a",
		Prompt:    "hi",
		Router:    defaultRouter(prov),
	})
	if err == nil {
		t.Fatal("expected error from Run, got nil")
	}
	if !obs.wasSeen() {
		t.Fatal("LLMObserver was not called on error path")
	}
	obs.mu.Lock()
	defer obs.mu.Unlock()
	if obs.err == nil {
		t.Error("observer err is nil, expected provider error")
	}
}

// TestToolObserverCalled: observer records ToolCall.ToolName; assert it matches dispatched tool.
func TestToolObserverCalled(t *testing.T) {
	toolName := "my_tool"
	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse(toolName, "id1", []byte(`{}`)),
		sleipnirtest.TextResponse("done"),
	)
	obs := &captureToolObserver{}

	h := mustNewHarness(t, sleipnir.Config{
		Middlewares: []sleipnir.Middleware{obs},
	})
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name:         "a",
		MaxIterations: 5,
		Tools:        []sleipnir.Tool{sleipnirtest.StaticTool(toolName, "result")},
	}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}
	_, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "a",
		Prompt:    "hi",
		Router:    defaultRouter(stub),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	names := obs.names()
	if len(names) == 0 {
		t.Fatal("ToolObserver was never called")
	}
	if names[0] != toolName {
		t.Errorf("expected tool name %q, got %q", toolName, names[0])
	}
}

// TestRetryPolicyRetries: policy always retries with zero backoff; provider fails once then succeeds;
// total LLM calls == 2.
func TestRetryPolicyRetries(t *testing.T) {
	provErr := errors.New("transient error")
	prov := &failThenSucceedProvider{failFor: 1, err: provErr}
	policy := &alwaysRetryPolicy{}

	h := mustNewHarness(t, sleipnir.Config{
		MaxLLMRetries: 3,
		Middlewares:   []sleipnir.Middleware{policy},
	})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "a", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}
	_, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "a",
		Prompt:    "hi",
		Router:    defaultRouter(prov),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	calls := prov.calls.Load()
	if calls != 2 {
		t.Errorf("expected 2 LLM calls (1 fail + 1 success), got %d", calls)
	}
}

// TestRetryPolicyNoRetry: policy returns (false, 0); provider fails → error propagates; total calls == 1.
func TestRetryPolicyNoRetry(t *testing.T) {
	provErr := errors.New("fatal error")
	prov := &errorProvider{err: provErr}
	policy := &neverRetryPolicy{}

	h := mustNewHarness(t, sleipnir.Config{
		MaxLLMRetries: 3,
		Middlewares:   []sleipnir.Middleware{policy},
	})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "a", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}
	_, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "a",
		Prompt:    "hi",
		Router:    defaultRouter(prov),
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	calls := prov.calls.Load()
	if calls != 1 {
		t.Errorf("expected 1 LLM call, got %d", calls)
	}
}

// TestRetryMaxExhausted: MaxLLMRetries: 2, policy always retries;
// provider always fails → LLM called exactly 3 times (attempts 0, 1, 2).
func TestRetryMaxExhausted(t *testing.T) {
	provErr := errors.New("always fails")
	prov := &errorProvider{err: provErr}
	policy := &alwaysRetryPolicy{}

	h := mustNewHarness(t, sleipnir.Config{
		MaxLLMRetries: 2,
		Middlewares:   []sleipnir.Middleware{policy},
	})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "a", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}
	_, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "a",
		Prompt:    "hi",
		Router:    defaultRouter(prov),
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	calls := prov.calls.Load()
	if calls != 3 {
		t.Errorf("expected 3 LLM calls (attempts 0, 1, 2), got %d", calls)
	}
}

// TestEffectiveMiddlewaresAgentOverrides: agent spec has non-nil Middlewares with observer A;
// config has observer B; only A is called.
func TestEffectiveMiddlewaresAgentOverrides(t *testing.T) {
	stub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("done"))
	obsA := &captureLLMObserver{}
	obsB := &captureLLMObserver{}

	h := mustNewHarness(t, sleipnir.Config{
		Middlewares: []sleipnir.Middleware{obsB},
	})
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name:          "a",
		MaxIterations: 5,
		Middlewares:   []sleipnir.Middleware{obsA},
	}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}
	_, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "a",
		Prompt:    "hi",
		Router:    defaultRouter(stub),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !obsA.wasSeen() {
		t.Error("observer A (agent-level) was not called")
	}
	if obsB.wasSeen() {
		t.Error("observer B (config-level) was called despite agent override")
	}
}

// TestEffectiveMiddlewaresConfigFallback: agent spec Middlewares is nil; config has observer B; B is called.
func TestEffectiveMiddlewaresConfigFallback(t *testing.T) {
	stub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("done"))
	obsB := &captureLLMObserver{}

	h := mustNewHarness(t, sleipnir.Config{
		Middlewares: []sleipnir.Middleware{obsB},
	})
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name:          "a",
		MaxIterations: 5,
		// Middlewares is nil → falls back to config
	}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}
	_, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "a",
		Prompt:    "hi",
		Router:    defaultRouter(stub),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !obsB.wasSeen() {
		t.Error("observer B (config-level) was not called despite nil agent Middlewares")
	}
}

// TestEffectiveMiddlewaresAgentEmptyOverride: agent spec Middlewares is []Middleware{} (empty non-nil);
// config has observer B; B is NOT called.
func TestEffectiveMiddlewaresAgentEmptyOverride(t *testing.T) {
	stub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("done"))
	obsB := &captureLLMObserver{}

	h := mustNewHarness(t, sleipnir.Config{
		Middlewares: []sleipnir.Middleware{obsB},
	})
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name:          "a",
		MaxIterations: 5,
		Middlewares:   []sleipnir.Middleware{}, // empty non-nil overrides config
	}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}
	_, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "a",
		Prompt:    "hi",
		Router:    defaultRouter(stub),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if obsB.wasSeen() {
		t.Error("observer B (config-level) was called despite empty non-nil agent Middlewares")
	}
}

// TestMultipleObserversAllCalled: two LLMObserver implementations in the chain; both called in order.
func TestMultipleObserversAllCalled(t *testing.T) {
	stub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("done"))
	obsA := &captureLLMObserver{}
	obsB := &captureLLMObserver{}

	h := mustNewHarness(t, sleipnir.Config{
		Middlewares: []sleipnir.Middleware{obsA, obsB},
	})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "a", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}
	_, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "a",
		Prompt:    "hi",
		Router:    defaultRouter(stub),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !obsA.wasSeen() {
		t.Error("observer A was not called")
	}
	if !obsB.wasSeen() {
		t.Error("observer B was not called")
	}
}
