package sleipnir

import (
	"context"
	"encoding/json"
	"time"
)

// Middleware is a marker interface for middleware values. A middleware may
// implement any subset of ContextRewriter, LLMObserver, ToolObserver, and
// RetryPolicy. The harness discovers capabilities via type assertion.
type Middleware interface{ middleware() }

// ContextRewriter can mutate an LLMRequest immediately before it is sent to
// the provider. Mutations are visible to the provider but not persisted to the
// conversation history. Returning a non-nil error emits an ErrorEvent; the
// request is sent as-is.
type ContextRewriter interface {
	Middleware
	RewriteBeforeLLMCall(ctx context.Context, req *LLMRequest) error
}

// LLMObserver is called after every provider call, whether it succeeds or
// fails. It must not mutate req or resp.
type LLMObserver interface {
	Middleware
	OnLLMCall(ctx context.Context, req *LLMRequest, resp *LLMResponse, err error)
}

// ToolObserver is called after every tool invocation, including sub-agent
// calls. It must not mutate tc or result.
type ToolObserver interface {
	Middleware
	OnToolCall(ctx context.Context, call *ToolCall, result *ToolResult, err error)
}

// RetryPolicy decides whether a failed provider call should be retried and how
// long to wait before the next attempt. The first RetryPolicy in the chain
// wins; subsequent policies are not consulted.
type RetryPolicy interface {
	Middleware
	ShouldRetry(ctx context.Context, attempt int, err error) (retry bool, backoff time.Duration)
}

// ToolCall carries the agent identity and call parameters for ToolObserver.
// Middleware must not mutate Agent fields.
type ToolCall struct {
	Agent      AgentInfo
	ToolCallID string
	ToolName   string
	Args       json.RawMessage
}

// BaseMiddleware can be embedded in any struct to satisfy the [Middleware]
// marker interface. External packages (e.g. sleipnirtest, application code)
// must embed this to produce valid Middleware values.
type BaseMiddleware struct{}

func (BaseMiddleware) middleware() {}
