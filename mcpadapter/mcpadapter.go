// Package mcpadapter bridges the MCP SDK with Sleipnir tools.
// It queries an MCP server's tool list and wraps each tool as a sleipnir.Tool
// that can be passed to RunInput.ExtraTools.
package mcpadapter

import (
	"context"
	"encoding/json"
	"fmt"

	mcp "github.com/modelcontextprotocol/go-sdk/mcp"
	sleipnir "sleipnir.dev/sleipnir"
)

type loadConfig struct {
	prefix string
}

// Option configures LoadTools behaviour.
type Option func(*loadConfig)

// WithPrefix prepends a string to every tool name loaded from the MCP server.
func WithPrefix(prefix string) Option {
	return func(cfg *loadConfig) { cfg.prefix = prefix }
}

func applyOptions(opts []Option) loadConfig {
	var cfg loadConfig
	for _, o := range opts {
		o(&cfg)
	}
	return cfg
}

// LoadTools queries the MCP server for its tool list and wraps each as a
// sleipnir.Tool. The list is snapshotted at call time; call again to refresh.
func LoadTools(ctx context.Context, client *mcp.ClientSession, opts ...Option) ([]sleipnir.Tool, error) {
	cfg := applyOptions(opts)

	resp, err := client.ListTools(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("mcpadapter: list tools: %w", err)
	}

	tools := make([]sleipnir.Tool, len(resp.Tools))
	for i, mt := range resp.Tools {
		name := cfg.prefix + mt.Name

		// Convert MCP InputSchema (any) to map[string]any.
		var schema map[string]any
		if mt.InputSchema != nil {
			b, err := json.Marshal(mt.InputSchema)
			if err == nil {
				_ = json.Unmarshal(b, &schema)
			}
		}
		if schema == nil {
			schema = map[string]any{"type": "object"}
		}

		tools[i] = sleipnir.NewFuncTool(name, mt.Description, schema,
			func(ctx context.Context, args json.RawMessage) (sleipnir.ToolResult, error) {
				// Decode raw JSON args into a map for the MCP Arguments field.
				var argMap map[string]any
				if len(args) > 0 {
					if err := json.Unmarshal(args, &argMap); err != nil {
						return sleipnir.ToolResult{}, fmt.Errorf("mcpadapter: unmarshal args for %q: %w", mt.Name, err)
					}
				}

				// Use original mt.Name on the wire; the prefixed name is only for the LLM.
				result, err := client.CallTool(ctx, &mcp.CallToolParams{
					Name:      mt.Name,
					Arguments: argMap,
				})
				if err != nil {
					return sleipnir.ToolResult{}, fmt.Errorf("mcpadapter: call %q: %w", mt.Name, err)
				}

				// Extract the first text content from the result.
				var content string
				for _, c := range result.Content {
					if tc, ok := c.(*mcp.TextContent); ok {
						content = tc.Text
						break
					}
				}
				return sleipnir.ToolResult{
					Content: content,
					IsError: result.IsError,
				}, nil
			},
		)
	}
	return tools, nil
}
