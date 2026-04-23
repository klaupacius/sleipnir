package sleipnir

import (
	"context"
	"encoding/json"
	"fmt"

	"sleipnir.dev/sleipnir/internal/schema"
)

type typedTool[T any] struct {
	def ToolDefinition
	fn  func(context.Context, T) (ToolResult, error)
}

func (t *typedTool[T]) Definition() ToolDefinition { return t.def }

func (t *typedTool[T]) Invoke(ctx context.Context, args json.RawMessage) (ToolResult, error) {
	var input T
	if err := json.Unmarshal(args, &input); err != nil {
		return ToolResult{
			IsError: true,
			Content: fmt.Sprintf("invalid arguments: %s", err),
		}, nil
	}
	return t.fn(ctx, input)
}

func NewTypedTool[T any](name, desc string, fn func(context.Context, T) (ToolResult, error)) (Tool, error) {
	inputSchema, err := schema.Reflect[T]()
	if err != nil {
		return nil, fmt.Errorf("NewTypedTool %q: %w", name, err)
	}
	return &typedTool[T]{
		def: ToolDefinition{Name: name, Description: desc, InputSchema: inputSchema},
		fn:  fn,
	}, nil
}

var _ Tool = (*typedTool[struct{}])(nil)
