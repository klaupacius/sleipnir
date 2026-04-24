package accounting

import (
	"context"
	"sync"
	"testing"

	sleipnir "sleipnir.dev/sleipnir"
)

func makeReq(agentName string) *sleipnir.LLMRequest {
	return &sleipnir.LLMRequest{Agent: sleipnir.AgentInfo{Name: agentName}}
}

func makeResp(in, out, total int64) *sleipnir.LLMResponse {
	return &sleipnir.LLMResponse{Usage: sleipnir.Usage{
		InputTokens:  in,
		OutputTokens: out,
		TotalTokens:  total,
	}}
}

func TestTokenAccountantSingleAgent(t *testing.T) {
	a := &TokenAccountant{}
	ctx := context.Background()

	a.OnLLMCall(ctx, makeReq("alice"), makeResp(10, 5, 15), nil)
	a.OnLLMCall(ctx, makeReq("alice"), makeResp(20, 10, 30), nil)

	u := a.ByAgent("alice")
	if u.InputTokens != 30 {
		t.Errorf("InputTokens: want 30, got %d", u.InputTokens)
	}
	if u.OutputTokens != 15 {
		t.Errorf("OutputTokens: want 15, got %d", u.OutputTokens)
	}
	if u.TotalTokens != 45 {
		t.Errorf("TotalTokens: want 45, got %d", u.TotalTokens)
	}
}

func TestTokenAccountantMultiAgent(t *testing.T) {
	a := &TokenAccountant{}
	ctx := context.Background()

	a.OnLLMCall(ctx, makeReq("a"), makeResp(10, 5, 15), nil)
	a.OnLLMCall(ctx, makeReq("b"), makeResp(20, 10, 30), nil)

	ua := a.ByAgent("a")
	if ua.TotalTokens != 15 {
		t.Errorf("agent a TotalTokens: want 15, got %d", ua.TotalTokens)
	}

	ub := a.ByAgent("b")
	if ub.TotalTokens != 30 {
		t.Errorf("agent b TotalTokens: want 30, got %d", ub.TotalTokens)
	}

	total := a.Total()
	if total.TotalTokens != 45 {
		t.Errorf("Total TotalTokens: want 45, got %d", total.TotalTokens)
	}
	if total.InputTokens != 30 {
		t.Errorf("Total InputTokens: want 30, got %d", total.InputTokens)
	}
	if total.OutputTokens != 15 {
		t.Errorf("Total OutputTokens: want 15, got %d", total.OutputTokens)
	}
}

func TestTokenAccountantIgnoresErrors(t *testing.T) {
	a := &TokenAccountant{}
	ctx := context.Background()

	// Call with error — should not be counted
	a.OnLLMCall(ctx, makeReq("alice"), makeResp(100, 50, 150), errFake)

	u := a.ByAgent("alice")
	if u.TotalTokens != 0 {
		t.Errorf("expected 0 tokens after failed call, got %d", u.TotalTokens)
	}
}

var errFake = &fakeError{}

type fakeError struct{}

func (*fakeError) Error() string { return "fake error" }

func TestTokenAccountantConcurrentSafe(t *testing.T) {
	a := &TokenAccountant{}
	ctx := context.Background()

	const goroutines = 10
	const callsPerGoroutine = 100

	var wg sync.WaitGroup
	wg.Add(goroutines)
	for range goroutines {
		go func() {
			defer wg.Done()
			for range callsPerGoroutine {
				a.OnLLMCall(ctx, makeReq("agent"), makeResp(1, 1, 2), nil)
			}
		}()
	}
	wg.Wait()

	total := a.Total()
	expected := int64(goroutines * callsPerGoroutine * 2)
	if total.TotalTokens != expected {
		t.Errorf("Total TotalTokens: want %d, got %d", expected, total.TotalTokens)
	}
}
