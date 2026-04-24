package sleipnir

import (
	"context"
	"encoding/json"
	"errors"
	"time"
)

func runStateFromCtx(ctx context.Context) *runState {
	rs, _ := ctx.Value(runStateCtxKey{}).(*runState)
	return rs // nil if not set; tools handle nil gracefully
}

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

// TodoWriteTool returns a Tool that replaces the current todo list for the
// invoking agent and emits a TodoEvent. The list is keyed by agent name:
// state persists across turns, but two concurrent calls from the same agent
// are not isolated from each other. Pass via RunInput.ExtraTools.
func TodoWriteTool() Tool {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"tasks": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"id":     map[string]any{"type": "string"},
						"text":   map[string]any{"type": "string"},
						"status": map[string]any{"type": "string", "enum": []string{"pending", "in_progress", "done"}},
					},
					"required": []string{"id", "text", "status"},
				},
			},
		},
		"required": []string{"tasks"},
	}
	return NewFuncTool("todo_write", "Replace the current todo list.", schema,
		func(ctx context.Context, args json.RawMessage) (ToolResult, error) {
			var input struct {
				Tasks []TodoItem `json:"tasks"`
			}
			if err := json.Unmarshal(args, &input); err != nil {
				return ToolResult{IsError: true, Content: "invalid todo_write arguments: " + err.Error()}, nil
			}
			rs := runStateFromCtx(ctx)
			if rs == nil {
				return ToolResult{IsError: true, Content: "todo_write: no run state in context"}, nil
			}
			agentName := agentNameFromCtx(ctx)
			callID := toolCallIDFromCtx(ctx)
			rs.todos.Store(agentName, input.Tasks)

			sink := sinkFromCtx(ctx)
			emit(sink, TodoEvent{
				AgentName:  agentName,
				ToolCallID: callID,
				Todos:      input.Tasks,
			})
			return ToolResult{Content: "todo list updated"}, nil
		},
	)
}

// TodoReadTool returns a Tool that reads the current todo list for the invoking
// agent. Returns "[]" if no list has been written yet.
// Pass via RunInput.ExtraTools.
func TodoReadTool() Tool {
	return NewFuncTool("todo_read", "Read the current todo list.", map[string]any{"type": "object"},
		func(ctx context.Context, _ json.RawMessage) (ToolResult, error) {
			rs := runStateFromCtx(ctx)
			if rs == nil {
				return ToolResult{Content: "[]"}, nil
			}
			agentName := agentNameFromCtx(ctx)
			v, _ := rs.todos.Load(agentName)
			items, _ := v.([]TodoItem)
			if items == nil {
				items = []TodoItem{}
			}
			data, _ := json.Marshal(items)
			return ToolResult{Content: string(data)}, nil
		},
	)
}
