package sleipnirtest

import (
	"context"
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func TestFakeMCPServerRoundTrip(t *testing.T) {
	f := NewFakeMCPServer(t)

	f.AddTool("greet", "Say hello", func(ctx context.Context, args map[string]any) (string, error) {
		name, _ := args["name"].(string)
		if name == "" {
			name = "world"
		}
		return "Hello, " + name + "!", nil
	})

	f.Start()

	cs := f.Client()
	result, err := cs.CallTool(context.Background(), &mcp.CallToolParams{
		Name:      "greet",
		Arguments: map[string]any{"name": "sleipnir"},
	})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if result.IsError {
		t.Fatalf("tool returned error: %v", result.Content)
	}
	if len(result.Content) == 0 {
		t.Fatal("expected non-empty content")
	}
	text, ok := result.Content[0].(*mcp.TextContent)
	if !ok {
		t.Fatalf("expected *mcp.TextContent, got %T", result.Content[0])
	}
	if !strings.Contains(text.Text, "sleipnir") {
		t.Errorf("expected response to contain 'sleipnir', got %q", text.Text)
	}
}
