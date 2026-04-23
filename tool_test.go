package sleipnir

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// MockTool implements the Tool interface for testing
type MockTool struct {
	name        string
	description string
	inputSchema map[string]any
	invokeFunc  func(ctx context.Context, args json.RawMessage) (ToolResult, error)
}

func (m *MockTool) Definition() ToolDefinition {
	return ToolDefinition{
		Name:        m.name,
		Description: m.description,
		InputSchema: m.inputSchema,
	}
}

func (m *MockTool) Invoke(ctx context.Context, args json.RawMessage) (ToolResult, error) {
	if m.invokeFunc != nil {
		return m.invokeFunc(ctx, args)
	}
	return ToolResult{Content: "mock result", IsError: false}, nil
}

func TestValidateToolNamesEmpty(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name  string
		tools []Tool
	}{
		{"nil slice", nil},
		{"empty slice", []Tool{}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateToolNames(tc.tools)
			if err != nil {
				t.Errorf("Expected nil error, got %v", err)
			}
		})
	}
}

func TestValidateToolNamesUnique(t *testing.T) {
	t.Parallel()

	tools := []Tool{
		&MockTool{name: "tool1"},
		&MockTool{name: "tool2"},
		&MockTool{name: "tool3"},
	}

	err := validateToolNames(tools)
	if err != nil {
		t.Errorf("Expected nil error for distinct tool names, got %v", err)
	}
}

func TestValidateToolNamesDuplicate(t *testing.T) {
	t.Parallel()

	tools := []Tool{
		&MockTool{name: "calculator"},
		&MockTool{name: "search"},
		&MockTool{name: "calculator"}, // duplicate name
	}

	err := validateToolNames(tools)

	if err == nil {
		t.Error("Expected error for duplicate tool names, got nil")
	}

	if !errors.Is(err, ErrToolNameCollision) {
		t.Errorf("Expected error to wrap ErrToolNameCollision, got %v", err)
	}
}

func TestValidateToolNamesFirstDuplicate(t *testing.T) {
	t.Parallel()

	duplicateName := "duplicate_tool"

	tools := []Tool{
		&MockTool{name: "first"},
		&MockTool{name: duplicateName},
		&MockTool{name: "second"},
		&MockTool{name: duplicateName}, // duplicate appears again
	}

	err := validateToolNames(tools)

	if err == nil {
		t.Fatal("Expected error for duplicate tool names, got nil")
	}

	// Check that the error message includes the colliding name
	if !errors.Is(err, ErrToolNameCollision) {
		t.Errorf("Expected error to wrap ErrToolNameCollision, got %v", err)
	}

	// Check that the error message contains the duplicate name
	errMsg := err.Error()
	if !strings.Contains(errMsg, duplicateName) {
		t.Errorf("Expected error message to include colliding name %q, got %q", duplicateName, errMsg)
	}
}
