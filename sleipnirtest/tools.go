package sleipnirtest

import (
	"context"
	"encoding/json"

	sleipnir "sleipnir.dev/sleipnir"
)

type staticTool struct {
	name   string
	result string
}

func (s staticTool) Definition() sleipnir.ToolDefinition {
	return sleipnir.ToolDefinition{Name: s.name}
}

func (s staticTool) Invoke(_ context.Context, _ json.RawMessage) (sleipnir.ToolResult, error) {
	return sleipnir.ToolResult{Content: s.result}, nil
}

// StaticTool returns a Tool with the given name that always returns result.
// Useful for registering tools with duplicate names to trigger ErrToolNameCollision.
func StaticTool(name, result string) sleipnir.Tool {
	return staticTool{name: name, result: result}
}
