package sleipnir

import (
	"context"
	"encoding/json"
	"errors"
	"time"
)

// Context helper functions for built-in tools.

func sinkFromCtx(ctx context.Context) Sink {
	s, _ := ctx.Value(sinkCtxKey{}).(Sink)
	return s
}

func agentNameFromCtx(ctx context.Context) string {
	n, _ := ctx.Value(agentNameCtxKey{}).(string)
	return n
}

func toolCallIDFromCtx(ctx context.Context) string {
	id, _ := ctx.Value(toolCallIDCtxKey{}).(string)
	return id
}

func hitlTimeoutFromCtx(ctx context.Context) time.Duration {
	d, _ := ctx.Value(hitlTimeoutCtxKey{}).(time.Duration)
	if d <= 0 {
		return 5 * time.Minute // default
	}
	return d
}

// AskUserTool returns a Tool that, when invoked, emits a QuestionEvent and
// then delegates to the provided HITLHandler. Pass via RunInput.ExtraTools.
// If handler is nil, Invoke returns ToolResult{IsError: true} immediately.
func AskUserTool(handler HITLHandler) Tool {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"question": map[string]any{"type": "string"},
			"context":  map[string]any{"type": "string"},
		},
		"required": []string{"question"},
	}
	return NewFuncTool("ask_user", "Ask the user a question and wait for their reply.", schema,
		func(ctx context.Context, args json.RawMessage) (ToolResult, error) {
			if handler == nil {
				return ToolResult{IsError: true, Content: "HITL handler not configured"}, nil
			}

			var input struct {
				Question string `json:"question"`
				Context  string `json:"context"`
			}
			if err := json.Unmarshal(args, &input); err != nil {
				return ToolResult{IsError: true, Content: "invalid ask_user arguments: " + err.Error()}, nil
			}

			questionID := toolCallIDFromCtx(ctx)
			sink := sinkFromCtx(ctx)
			emit(sink, QuestionEvent{
				AgentName:  agentNameFromCtx(ctx),
				QuestionID: questionID,
				Question:   input.Question,
			})

			hitlCtx, cancel := context.WithTimeout(ctx, hitlTimeoutFromCtx(ctx))
			defer cancel()

			reply, err := handler.AskUser(hitlCtx, agentNameFromCtx(ctx), input.Question, input.Context)
			if err != nil {
				if errors.Is(err, context.DeadlineExceeded) {
					return ToolResult{}, ErrHITLTimeout
				}
				if errors.Is(err, context.Canceled) {
					return ToolResult{}, ErrHITLCancelled
				}
				return ToolResult{IsError: true, Content: "ask_user failed: " + err.Error()}, nil
			}
			return ToolResult{Content: reply}, nil
		},
	)
}
