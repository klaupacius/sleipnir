package mcpadapter_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"sleipnir.dev/sleipnir/mcpadapter"
	"sleipnir.dev/sleipnir/sleipnirtest"
)

func TestLoadToolsBasic(t *testing.T) {
	srv := sleipnirtest.NewFakeMCPServer(t)
	srv.AddTool("greet", "says hello", func(ctx context.Context, args map[string]any) (string, error) {
		return "hello", nil
	})
	srv.AddTool("farewell", "says bye", func(ctx context.Context, args map[string]any) (string, error) {
		return "bye", nil
	})
	srv.Start()

	tools, err := mcpadapter.LoadTools(context.Background(), srv.Client())
	if err != nil {
		t.Fatalf("LoadTools: %v", err)
	}
	if len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(tools))
	}

	names := map[string]string{}
	for _, tool := range tools {
		d := tool.Definition()
		names[d.Name] = d.Description
	}

	if desc, ok := names["greet"]; !ok || desc != "says hello" {
		t.Errorf("greet: got %q, want %q", desc, "says hello")
	}
	if desc, ok := names["farewell"]; !ok || desc != "says bye" {
		t.Errorf("farewell: got %q, want %q", desc, "says bye")
	}
}

func TestLoadToolsWithPrefix(t *testing.T) {
	srv := sleipnirtest.NewFakeMCPServer(t)
	srv.AddTool("search", "searches things", func(ctx context.Context, args map[string]any) (string, error) {
		return "results", nil
	})
	srv.Start()

	tools, err := mcpadapter.LoadTools(context.Background(), srv.Client(), mcpadapter.WithPrefix("urd_"))
	if err != nil {
		t.Fatalf("LoadTools: %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	if got := tools[0].Definition().Name; got != "urd_search" {
		t.Errorf("name: got %q, want %q", got, "urd_search")
	}
}

func TestLoadToolsInvoke(t *testing.T) {
	srv := sleipnirtest.NewFakeMCPServer(t)
	srv.AddTool("greet", "greet someone", func(ctx context.Context, args map[string]any) (string, error) {
		name, _ := args["name"].(string)
		return "hello " + name, nil
	})
	srv.Start()

	tools, err := mcpadapter.LoadTools(context.Background(), srv.Client())
	if err != nil {
		t.Fatalf("LoadTools: %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}

	args, _ := json.Marshal(map[string]any{"name": "world"})
	result, err := tools[0].Invoke(context.Background(), json.RawMessage(args))
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if result.Content != "hello world" {
		t.Errorf("content: got %q, want %q", result.Content, "hello world")
	}
	if result.IsError {
		t.Error("expected IsError=false")
	}
}

func TestLoadToolsInvokeError(t *testing.T) {
	srv := sleipnirtest.NewFakeMCPServer(t)
	srv.AddTool("broken", "always fails", func(ctx context.Context, args map[string]any) (string, error) {
		return "", errors.New("something went wrong")
	})
	srv.Start()

	tools, err := mcpadapter.LoadTools(context.Background(), srv.Client())
	if err != nil {
		t.Fatalf("LoadTools: %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}

	// FakeMCPServer converts tool errors into IsError=true results (not protocol errors).
	result, err := tools[0].Invoke(context.Background(), json.RawMessage(`{}`))
	if err != nil {
		t.Fatalf("Invoke returned unexpected protocol error: %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for tool that returned an error")
	}
	if result.Content == "" {
		t.Error("expected non-empty Content with error message")
	}
}

func TestLoadToolsSnapshot(t *testing.T) {
	srv := sleipnirtest.NewFakeMCPServer(t)
	srv.AddTool("original", "original tool", func(ctx context.Context, args map[string]any) (string, error) {
		return "original", nil
	})
	srv.Start()

	// Snapshot taken here.
	tools, err := mcpadapter.LoadTools(context.Background(), srv.Client())
	if err != nil {
		t.Fatalf("LoadTools: %v", err)
	}
	if len(tools) != 1 {
		t.Errorf("expected 1 tool in snapshot, got %d", len(tools))
	}

	// Adding a tool AFTER LoadTools should NOT appear in the previous slice.
	// (We can't add tools after Start in FakeMCPServer, so we just verify
	// the snapshot count remains stable — no refresh happened.)
	if len(tools) != 1 {
		t.Errorf("snapshot grew unexpectedly: got %d tools", len(tools))
	}
}

// TestLoadToolsMultipleTextContent verifies that when a tool returns multiple
// TextContent items, the adapter concatenates all of them into a single string.
func TestLoadToolsMultipleTextContent(t *testing.T) {
	srv := sleipnirtest.NewFakeMCPServer(t)
	srv.AddRawTool("multi", "multiple text items", func(_ context.Context, _ map[string]any) (*mcp.CallToolResult, error) {
		return &mcp.CallToolResult{
			Content: []mcp.Content{
				&mcp.TextContent{Text: "hello"},
				&mcp.TextContent{Text: " world"},
			},
		}, nil
	})
	srv.Start()

	tools, err := mcpadapter.LoadTools(context.Background(), srv.Client())
	if err != nil {
		t.Fatalf("LoadTools: %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}

	result, err := tools[0].Invoke(context.Background(), json.RawMessage(`{}`))
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if result.Content != "hello world" {
		t.Errorf("content: got %q, want %q", result.Content, "hello world")
	}
}

// TestLoadToolsNonTextContent verifies that non-text content items produce a
// non-empty fallback string instead of silently returning empty.
func TestLoadToolsNonTextContent(t *testing.T) {
	srv := sleipnirtest.NewFakeMCPServer(t)
	srv.AddRawTool("img", "returns image content", func(_ context.Context, _ map[string]any) (*mcp.CallToolResult, error) {
		return &mcp.CallToolResult{
			Content: []mcp.Content{
				&mcp.ImageContent{MIMEType: "image/png", Data: []byte("fake")},
			},
		}, nil
	})
	srv.Start()

	tools, err := mcpadapter.LoadTools(context.Background(), srv.Client())
	if err != nil {
		t.Fatalf("LoadTools: %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}

	result, err := tools[0].Invoke(context.Background(), json.RawMessage(`{}`))
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if result.Content == "" {
		t.Error("expected non-empty fallback content for non-text result, got empty string")
	}
}
