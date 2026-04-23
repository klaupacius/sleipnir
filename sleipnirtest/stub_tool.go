package sleipnirtest

import (
	"context"
	"encoding/json"
	"sync"

	"sleipnir.dev/sleipnir"
)

type StubToolInvocation struct {
	Args   json.RawMessage
	Result sleipnir.ToolResult
	Err    error
}

// StubTool is a Tool for use in tests. It records every invocation.
type StubTool struct {
	name        string
	desc        string
	schema      map[string]any
	fn          func(context.Context, json.RawMessage) (sleipnir.ToolResult, error)
	mu          sync.Mutex
	invocations []StubToolInvocation
}

func NewStubTool(name, desc string, schema map[string]any,
	fn func(context.Context, json.RawMessage) (sleipnir.ToolResult, error)) *StubTool {
	return &StubTool{name: name, desc: desc, schema: schema, fn: fn}
}

func (s *StubTool) Definition() sleipnir.ToolDefinition {
	return sleipnir.ToolDefinition{Name: s.name, Description: s.desc, InputSchema: s.schema}
}

func (s *StubTool) Invoke(ctx context.Context, args json.RawMessage) (sleipnir.ToolResult, error) {
	res, err := s.fn(ctx, args)
	s.mu.Lock()
	s.invocations = append(s.invocations, StubToolInvocation{Args: args, Result: res, Err: err})
	s.mu.Unlock()
	return res, err
}

func (s *StubTool) Invocations() []StubToolInvocation {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]StubToolInvocation, len(s.invocations))
	copy(out, s.invocations)
	return out
}

func (s *StubTool) InvokeCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.invocations)
}

var _ sleipnir.Tool = (*StubTool)(nil)
