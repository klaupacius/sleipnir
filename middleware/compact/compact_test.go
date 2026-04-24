package compact_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	anyllm "github.com/mozilla-ai/any-llm-go"
	sleipnir "sleipnir.dev/sleipnir"
	"sleipnir.dev/sleipnir/middleware/compact"
	"sleipnir.dev/sleipnir/sleipnirtest"
)

// errProvider is a test-only anyllm.Provider that always returns an error.
type errProvider struct{ err error }

func (e *errProvider) Name() string { return "err_provider" }
func (e *errProvider) Completion(_ context.Context, _ anyllm.CompletionParams) (*anyllm.ChatCompletion, error) {
	return nil, e.err
}
func (e *errProvider) CompletionStream(_ context.Context, _ anyllm.CompletionParams) (<-chan anyllm.ChatCompletionChunk, <-chan error) {
	panic("errProvider: CompletionStream not implemented")
}

// simpleStore is a CompactStore backed by a plain map (single-goroutine tests only).
type simpleStore struct {
	watermarks map[string]int
}

func newSimpleStore() *simpleStore {
	return &simpleStore{watermarks: make(map[string]int)}
}

func (s *simpleStore) GetWatermark(agentName string) int {
	return s.watermarks[agentName]
}

func (s *simpleStore) SetWatermark(agentName string, n int) {
	s.watermarks[agentName] = n
}

// bigMessages returns n messages each with `chars` characters of content.
// With estimateTokens = len / 4, each message contributes chars/4 tokens.
func bigMessages(n, chars int) []anyllm.Message {
	content := strings.Repeat("a", chars)
	msgs := make([]anyllm.Message, n)
	for i := range msgs {
		msgs[i] = anyllm.Message{Role: anyllm.RoleUser, Content: content}
	}
	return msgs
}

// withStore injects cs into ctx using the exported sleipnir.WithCompactStore.
func withStore(ctx context.Context, cs sleipnir.CompactStore) context.Context {
	return sleipnir.WithCompactStore(ctx, cs)
}

func agentReq(name string, isSub bool, msgs []anyllm.Message) *sleipnir.LLMRequest {
	return &sleipnir.LLMRequest{
		Agent:    sleipnir.AgentInfo{Name: name, IsSubAgent: isSub},
		Messages: msgs,
	}
}

// TestCompactorUnderThreshold verifies that messages below the threshold are
// not modified.
func TestCompactorUnderThreshold(t *testing.T) {
	// ContextWindow=1000, Threshold=0.75 → trigger at 750 tokens = 3000 chars.
	// We use 2 messages × 400 chars = 800 chars → 200 tokens (well under 750).
	stub := sleipnirtest.NewStubProvider(t) // no responses needed
	c := compact.NewCompactor(compact.Config{
		Provider:      stub,
		Model:         "stub",
		Threshold:     0.75,
		ContextWindow: 1000,
	})

	msgs := bigMessages(2, 400)
	req := agentReq("agent", false, msgs)

	cs := newSimpleStore()
	ctx := withStore(context.Background(), cs)

	if err := c.RewriteBeforeLLMCall(ctx, req); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(req.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(req.Messages))
	}
	// Watermark should remain 0.
	if cs.GetWatermark("agent") != 0 {
		t.Fatalf("watermark should be 0, got %d", cs.GetWatermark("agent"))
	}
}

// TestCompactorOverThreshold verifies that messages above the threshold are
// summarized and replaced with a single summary message.
func TestCompactorOverThreshold(t *testing.T) {
	// ContextWindow=1000, Threshold=0.75 → trigger at 750 tokens = 3000 chars.
	// 4 messages × 800 chars = 3200 chars → 800 tokens (≥ 750).
	stub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("summary text"))
	c := compact.NewCompactor(compact.Config{
		Provider:      stub,
		Model:         "stub",
		Threshold:     0.75,
		ContextWindow: 1000,
	})

	msgs := bigMessages(4, 800)
	req := agentReq("agent", false, msgs)

	cs := newSimpleStore()
	ctx := withStore(context.Background(), cs)

	if err := c.RewriteBeforeLLMCall(ctx, req); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// After compaction: 1 summary message + 0 remaining (watermark was 0, compacted all 4).
	if len(req.Messages) != 1 {
		t.Fatalf("expected 1 message after compaction, got %d", len(req.Messages))
	}
	if !strings.Contains(req.Messages[0].ContentString(), "summary text") {
		t.Fatalf("expected summary message to contain 'summary text', got %q", req.Messages[0].ContentString())
	}
}

// TestCompactorWatermarkAdvances verifies that the watermark advances so that
// already-summarized messages are not re-summarized on a second call.
func TestCompactorWatermarkAdvances(t *testing.T) {
	// ContextWindow=1000, Threshold=0.75 → trigger at 750 tokens.
	// 4 messages × 800 chars = 3200 chars → 800 tokens.
	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.TextResponse("summary1"),
		sleipnirtest.TextResponse("summary2"),
	)
	c := compact.NewCompactor(compact.Config{
		Provider:      stub,
		Model:         "stub",
		Threshold:     0.75,
		ContextWindow: 1000,
	})

	cs := newSimpleStore()
	ctx := withStore(context.Background(), cs)

	// First call — above threshold, should compact.
	msgs1 := bigMessages(4, 800)
	req1 := agentReq("agent", false, msgs1)
	if err := c.RewriteBeforeLLMCall(ctx, req1); err != nil {
		t.Fatalf("first call: unexpected error: %v", err)
	}
	// Watermark should advance by 4 (all messages compacted).
	if wm := cs.GetWatermark("agent"); wm != 4 {
		t.Fatalf("after first compact: expected watermark=4, got %d", wm)
	}

	// Second call — above threshold again, should compact once more with a
	// fresh set of big messages.
	msgs2 := bigMessages(4, 800)
	req2 := agentReq("agent", false, msgs2)
	if err := c.RewriteBeforeLLMCall(ctx, req2); err != nil {
		t.Fatalf("second call: unexpected error: %v", err)
	}
	// Summarizer was called exactly twice (once per above-threshold call).
	// The StubProvider will fatalf if called a third time, so if we reach here
	// the invariant holds.
}

// TestCompactorSkipsSubAgent verifies that sub-agent requests are never compacted.
func TestCompactorSkipsSubAgent(t *testing.T) {
	stub := sleipnirtest.NewStubProvider(t) // no responses expected
	c := compact.NewCompactor(compact.Config{
		Provider:      stub,
		Model:         "stub",
		Threshold:     0.75,
		ContextWindow: 1000,
	})

	msgs := bigMessages(4, 800) // above threshold
	req := agentReq("sub", true, msgs)

	cs := newSimpleStore()
	ctx := withStore(context.Background(), cs)

	if err := c.RewriteBeforeLLMCall(ctx, req); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(req.Messages) != 4 {
		t.Fatalf("expected messages unchanged (4), got %d", len(req.Messages))
	}
}

// TestCompactorSummarizerError verifies that a summarizer failure returns an
// error and leaves req.Messages unmodified.
func TestCompactorSummarizerError(t *testing.T) {
	sentinel := errors.New("summarizer down")
	c := compact.NewCompactor(compact.Config{
		Provider:      &errProvider{err: sentinel},
		Model:         "stub",
		Threshold:     0.75,
		ContextWindow: 1000,
	})

	msgs := bigMessages(4, 800) // above threshold
	origLen := len(msgs)
	req := agentReq("agent", false, msgs)

	cs := newSimpleStore()
	ctx := withStore(context.Background(), cs)

	err := c.RewriteBeforeLLMCall(ctx, req)
	if err == nil {
		t.Fatal("expected error from failing summarizer, got nil")
	}
	if !errors.Is(err, sentinel) {
		t.Fatalf("expected error to wrap sentinel, got: %v", err)
	}
	// Messages must be unmodified.
	if len(req.Messages) != origLen {
		t.Fatalf("messages should be unmodified, expected %d, got %d", origLen, len(req.Messages))
	}
	// Watermark must not advance.
	if cs.GetWatermark("agent") != 0 {
		t.Fatalf("watermark should remain 0 on error, got %d", cs.GetWatermark("agent"))
	}
}

// TestCompactorNoStoreInCtx verifies that a context without a CompactStore
// causes no panic and returns nil.
func TestCompactorNoStoreInCtx(t *testing.T) {
	stub := sleipnirtest.NewStubProvider(t) // no responses expected
	c := compact.NewCompactor(compact.Config{
		Provider:      stub,
		Model:         "stub",
		Threshold:     0.75,
		ContextWindow: 1000,
	})

	msgs := bigMessages(4, 800) // above threshold
	req := agentReq("agent", false, msgs)

	// Plain context — no CompactStore injected.
	err := c.RewriteBeforeLLMCall(context.Background(), req)
	if err != nil {
		t.Fatalf("expected nil error without store, got: %v", err)
	}
	if len(req.Messages) != 4 {
		t.Fatalf("messages should be unchanged without store, expected 4, got %d", len(req.Messages))
	}
}
