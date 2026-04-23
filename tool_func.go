package sleipnir

import (
	"context"
	"encoding/json"
)

type funcTool struct {
	def ToolDefinition
	fn  func(context.Context, json.RawMessage) (ToolResult, error)
}

func (t *funcTool) Definition() ToolDefinition { return t.def }
func (t *funcTool) Invoke(ctx context.Context, args json.RawMessage) (ToolResult, error) {
	return t.fn(ctx, args)
}

func NewFuncTool(name, desc string, schema map[string]any, fn func(context.Context, json.RawMessage) (ToolResult, error)) Tool {
	return &funcTool{
		def: ToolDefinition{Name: name, Description: desc, InputSchema: schema},
		fn:  fn,
	}
}
