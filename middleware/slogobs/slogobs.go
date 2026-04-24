package slogobs

import (
	"context"
	"log/slog"

	sleipnir "sleipnir.dev/sleipnir"
)

// SlogObserver logs LLM calls and tool calls via slog. Nil Logger falls back to slog.Default().
type SlogObserver struct {
	sleipnir.BaseMiddleware
	Logger *slog.Logger
}

func (o *SlogObserver) logger() *slog.Logger {
	if o.Logger != nil {
		return o.Logger
	}
	return slog.Default()
}

func (o *SlogObserver) OnLLMCall(_ context.Context, req *sleipnir.LLMRequest, resp *sleipnir.LLMResponse, err error) {
	l := o.logger()
	if err != nil {
		l.Error("llm call failed", "agent", req.Agent.Name, "depth", req.Agent.Depth, "error", err)
		return
	}
	l.Debug("llm call", "agent", req.Agent.Name, "depth", req.Agent.Depth,
		"input_tokens", resp.Usage.InputTokens, "output_tokens", resp.Usage.OutputTokens)
}

func (o *SlogObserver) OnToolCall(_ context.Context, call *sleipnir.ToolCall, result *sleipnir.ToolResult, err error) {
	l := o.logger()
	if result.IsError || err != nil {
		l.Warn("tool call error", "agent", call.Agent.Name, "depth", call.Agent.Depth,
			"tool", call.ToolName, "is_error", result.IsError)
		return
	}
	l.Debug("tool call", "agent", call.Agent.Name, "depth", call.Agent.Depth, "tool", call.ToolName)
}

var _ sleipnir.LLMObserver = (*SlogObserver)(nil)
var _ sleipnir.ToolObserver = (*SlogObserver)(nil)
