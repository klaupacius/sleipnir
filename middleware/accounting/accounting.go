package accounting

import (
	"context"
	"sync"

	sleipnir "sleipnir.dev/sleipnir"
)

// TokenAccountant is an LLMObserver that tracks per-agent token usage.
// Failed LLM calls are not counted.
type TokenAccountant struct {
	sleipnir.BaseMiddleware
	mu      sync.Mutex
	byAgent map[string]sleipnir.Usage
}

func (a *TokenAccountant) OnLLMCall(_ context.Context, req *sleipnir.LLMRequest, resp *sleipnir.LLMResponse, err error) {
	if err != nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.byAgent == nil {
		a.byAgent = make(map[string]sleipnir.Usage)
	}
	u := a.byAgent[req.Agent.Name]
	u.InputTokens += resp.Usage.InputTokens
	u.OutputTokens += resp.Usage.OutputTokens
	u.TotalTokens += resp.Usage.TotalTokens
	a.byAgent[req.Agent.Name] = u
}

func (a *TokenAccountant) ByAgent(name string) sleipnir.Usage {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.byAgent[name]
}

func (a *TokenAccountant) Total() sleipnir.Usage {
	a.mu.Lock()
	defer a.mu.Unlock()
	var total sleipnir.Usage
	for _, u := range a.byAgent {
		total.InputTokens += u.InputTokens
		total.OutputTokens += u.OutputTokens
		total.TotalTokens += u.TotalTokens
	}
	return total
}

var _ sleipnir.LLMObserver = (*TokenAccountant)(nil)
