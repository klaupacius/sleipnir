package slogobs

import (
	"context"
	"errors"
	"log/slog"
	"strings"
	"testing"

	sleipnir "sleipnir.dev/sleipnir"
)

func newTestObserver(buf *strings.Builder) *SlogObserver {
	h := slog.NewTextHandler(buf, &slog.HandlerOptions{Level: slog.LevelDebug})
	return &SlogObserver{Logger: slog.New(h)}
}

func TestSlogObserverLLMSuccess(t *testing.T) {
	var buf strings.Builder
	obs := newTestObserver(&buf)

	req := &sleipnir.LLMRequest{Agent: sleipnir.AgentInfo{Name: "myagent", Depth: 0}}
	resp := &sleipnir.LLMResponse{Usage: sleipnir.Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15}}

	obs.OnLLMCall(context.Background(), req, resp, nil)

	out := buf.String()
	if !strings.Contains(out, "DEBUG") {
		t.Errorf("expected DEBUG in output, got: %q", out)
	}
	if !strings.Contains(out, "myagent") {
		t.Errorf("expected agent name in output, got: %q", out)
	}
}

func TestSlogObserverLLMError(t *testing.T) {
	var buf strings.Builder
	obs := newTestObserver(&buf)

	req := &sleipnir.LLMRequest{Agent: sleipnir.AgentInfo{Name: "myagent", Depth: 0}}

	obs.OnLLMCall(context.Background(), req, nil, errors.New("provider down"))

	out := buf.String()
	if !strings.Contains(out, "ERROR") {
		t.Errorf("expected ERROR in output, got: %q", out)
	}
}

func TestSlogObserverToolSuccess(t *testing.T) {
	var buf strings.Builder
	obs := newTestObserver(&buf)

	call := &sleipnir.ToolCall{Agent: sleipnir.AgentInfo{Name: "myagent"}, ToolName: "mytool"}
	result := &sleipnir.ToolResult{IsError: false}

	obs.OnToolCall(context.Background(), call, result, nil)

	out := buf.String()
	if !strings.Contains(out, "DEBUG") {
		t.Errorf("expected DEBUG in output, got: %q", out)
	}
}

func TestSlogObserverToolError(t *testing.T) {
	var buf strings.Builder
	obs := newTestObserver(&buf)

	call := &sleipnir.ToolCall{Agent: sleipnir.AgentInfo{Name: "myagent"}, ToolName: "mytool"}
	result := &sleipnir.ToolResult{IsError: true}

	obs.OnToolCall(context.Background(), call, result, nil)

	out := buf.String()
	if !strings.Contains(out, "WARN") {
		t.Errorf("expected WARN in output, got: %q", out)
	}
}

func TestSlogObserverNilLogger(t *testing.T) {
	obs := &SlogObserver{Logger: nil}

	req := &sleipnir.LLMRequest{Agent: sleipnir.AgentInfo{Name: "myagent"}}
	resp := &sleipnir.LLMResponse{}

	// Should not panic — falls back to slog.Default()
	obs.OnLLMCall(context.Background(), req, resp, nil)
}
