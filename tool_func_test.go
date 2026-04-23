package sleipnir

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

var _ Tool = (*funcTool)(nil)

func TestFuncToolDefinition(t *testing.T) {
	t.Parallel()

	name := "test_tool"
	desc := "a test tool for verification"
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"input": map[string]any{"type": "string"},
		},
	}

	fn := func(ctx context.Context, args json.RawMessage) (ToolResult, error) {
		return ToolResult{}, nil
	}

	tool := NewFuncTool(name, desc, schema, fn)
	def := tool.Definition()

	if def.Name != name {
		t.Errorf("expected name %q, got %q", name, def.Name)
	}
	if def.Description != desc {
		t.Errorf("expected description %q, got %q", desc, def.Description)
	}
	if len(def.InputSchema) != len(schema) {
		t.Errorf("expected schema length %d, got %d", len(schema), len(def.InputSchema))
	}
	// Deep compare schema contents
	schemaJSON, _ := json.Marshal(schema)
	defSchemaJSON, _ := json.Marshal(def.InputSchema)
	if string(schemaJSON) != string(defSchemaJSON) {
		t.Errorf("expected schema %s, got %s", schemaJSON, defSchemaJSON)
	}
}

func TestFuncToolInvoke(t *testing.T) {
	t.Parallel()

	expectedResult := ToolResult{
		Content: "test result",
		IsError: false,
	}
	var receivedArgs json.RawMessage

	fn := func(ctx context.Context, args json.RawMessage) (ToolResult, error) {
		receivedArgs = args
		return expectedResult, nil
	}

	tool := NewFuncTool("test", "", nil, fn)
	inputArgs := json.RawMessage(`{"foo":"bar"}`)

	result, err := tool.Invoke(context.Background(), inputArgs)

	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
	if result.Content != expectedResult.Content {
		t.Errorf("expected content %q, got %q", expectedResult.Content, result.Content)
	}
	if result.IsError != expectedResult.IsError {
		t.Errorf("expected IsError %v, got %v", expectedResult.IsError, result.IsError)
	}
	if string(receivedArgs) != string(inputArgs) {
		t.Errorf("expected args %s, got %s", inputArgs, receivedArgs)
	}
}

func TestFuncToolInvokeError(t *testing.T) {
	t.Parallel()

	expectedErr := errors.New("infrastructure failure")
	fn := func(ctx context.Context, args json.RawMessage) (ToolResult, error) {
		return ToolResult{}, expectedErr
	}

	tool := NewFuncTool("test", "", nil, fn)
	result, err := tool.Invoke(context.Background(), nil)

	// Error should be passed through unchanged
	if err != expectedErr {
		t.Errorf("expected error %v, got %v", expectedErr, err)
	}
	// ToolResult should be zero value (ignored)
	if result.Content != "" {
		t.Errorf("expected empty content, got %q", result.Content)
	}
	if result.IsError {
		t.Errorf("expected IsError false, got true")
	}
}

func TestFuncToolInvokeIsError(t *testing.T) {
	t.Parallel()

	expectedResult := ToolResult{
		Content: "user provided invalid input: missing required field 'id'",
		IsError: true,
	}
	fn := func(ctx context.Context, args json.RawMessage) (ToolResult, error) {
		return expectedResult, nil
	}

	tool := NewFuncTool("test", "", nil, fn)
	result, err := tool.Invoke(context.Background(), nil)

	// No infra error
	if err != nil {
		t.Errorf("expected nil error, got %v", err)
	}
	// ToolResult with IsError=true should be returned as-is
	if result.Content != expectedResult.Content {
		t.Errorf("expected content %q, got %q", expectedResult.Content, result.Content)
	}
	if !result.IsError {
		t.Errorf("expected IsError true, got false")
	}
}
