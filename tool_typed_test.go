package sleipnir_test

import (
	"context"
	"encoding/json"
	"testing"

	"sleipnir.dev/sleipnir"
)

type searchInput struct {
	Query string `json:"query" jsonschema:"required"`
}

type chanToolInput struct {
	Ch chan int `json:"ch"`
}

func TestNewTypedToolDefinition(t *testing.T) {
	fn := func(_ context.Context, _ searchInput) (sleipnir.ToolResult, error) {
		return sleipnir.ToolResult{Content: "ok"}, nil
	}
	tool, err := sleipnir.NewTypedTool("search", "desc", fn)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	def := tool.Definition()
	if def.Name != "search" {
		t.Errorf("expected name 'search', got %q", def.Name)
	}
	if _, ok := def.InputSchema["properties"]; !ok {
		t.Error("expected InputSchema to contain 'properties'")
	}
}

func TestNewTypedToolInvoke(t *testing.T) {
	var got searchInput
	fn := func(_ context.Context, in searchInput) (sleipnir.ToolResult, error) {
		got = in
		return sleipnir.ToolResult{Content: "ok"}, nil
	}
	tool, err := sleipnir.NewTypedTool("search", "desc", fn)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	result, err := tool.Invoke(context.Background(), json.RawMessage(`{"query":"hello"}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.IsError {
		t.Errorf("unexpected IsError=true: %s", result.Content)
	}
	if got.Query != "hello" {
		t.Errorf("expected query 'hello', got %q", got.Query)
	}
}

func TestNewTypedToolInvokeInvalidJSON(t *testing.T) {
	fn := func(_ context.Context, _ searchInput) (sleipnir.ToolResult, error) {
		return sleipnir.ToolResult{Content: "should not reach"}, nil
	}
	tool, err := sleipnir.NewTypedTool("search", "desc", fn)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	result, err := tool.Invoke(context.Background(), json.RawMessage(`{bad json}`))
	if err != nil {
		t.Fatalf("expected nil error for bad JSON, got %v", err)
	}
	if !result.IsError {
		t.Error("expected IsError=true for invalid JSON")
	}
}

func TestNewTypedToolSchemaError(t *testing.T) {
	fn := func(_ context.Context, _ chanToolInput) (sleipnir.ToolResult, error) {
		return sleipnir.ToolResult{}, nil
	}
	_, err := sleipnir.NewTypedTool("bad", "desc", fn)
	if err == nil {
		t.Error("expected error for channel type, got nil")
	}
}
