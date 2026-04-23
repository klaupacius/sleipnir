package sleipnir

import (
	"context"
	"encoding/json"
	"fmt"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

// Tool defines the behaviour of an agent tool
//
// There are two distinct failure modes:
//   - Infrastructure failure the LLM cannot meaningfully interpret
//     -> Wrap error as ToolResult{IsError: true, Content: "tool execution failed: "+err.Error()}
//     -> Emit ErrorEvent, continue run
//   - Structured failure the LLM can read and react to
//     -> passed to LLM as tool_result: (ToolResult{IsError: true}, nil)
type Tool interface {
	Definition() ToolDefinition
	Invoke(ctx context.Context, args json.RawMessage) (ToolResult, error)
}

type ToolDefinition struct {
	Name        string
	Description string
	InputSchema map[string]any
}

type ToolResult struct {
	Content string
	IsError bool
}

func validateToolNames(tools []Tool) error {
	seen := make(map[string]struct{}, len(tools))
	for _, t := range tools {
		name := t.Definition().Name
		if _, exists := seen[name]; exists {
			return fmt.Errorf("%w: %q", ErrToolNameCollision, name)
		}
		seen[name] = struct{}{}
	}
	return nil
}

// Maps each Tools ToolDefinition to
// anyllm.Tool{Type: "function", Function: anyllm.Function{Name, Description,Parameters}}
func toolsToAnyllm(tools []Tool) []anyllm.Tool {
	var mappedTools []anyllm.Tool
	for _, t := range tools {
		mappedTools = append(mappedTools, anyllm.Tool{
			Type: "function",
			Function: anyllm.Function{
				Name:        t.Definition().Name,
				Description: t.Definition().Description,
				Parameters:  t.Definition().InputSchema,
			},
		})
	}
	return mappedTools
}
