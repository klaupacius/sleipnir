package sleipnirtest

import (
	"context"
	"encoding/json"
	"sync"
	"testing"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

type StubProvider struct {
	t         *testing.T
	responses []*anyllm.ChatCompletion
	mu        sync.Mutex
	index     int
}

func NewStubProvider(t *testing.T, responses ...*anyllm.ChatCompletion) *StubProvider {
	return &StubProvider{t: t, responses: responses}
}

func (s *StubProvider) Name() string {
	return "stub_provider"
}

// Completion implements anyllm.Provider
func (s *StubProvider) Completion(_ context.Context, _ anyllm.CompletionParams) (*anyllm.ChatCompletion, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.index >= len(s.responses) {
		s.t.Fatalf("StubProvider: no more scripted responses (called %d times, have %d)", s.index+1, len(s.responses))
	}
	resp := s.responses[s.index]
	s.index++
	return resp, nil
}

// CompletionStream implements anyllm.Provider - not used in tests; panics with "not implemented"
func (s *StubProvider) CompletionStream(_ context.Context, _ anyllm.CompletionParams) (<-chan anyllm.ChatCompletionChunk, <-chan error) {
	panic("StubProvider: CompletionStream not implemented")
}

// MultiToolCallResponse returns a *ChatCompletion scripted to call multiple tools in a single turn.
func MultiToolCallResponse(calls ...anyllm.ToolCall) *anyllm.ChatCompletion {
	return &anyllm.ChatCompletion{
		Choices: []anyllm.Choice{{
			Message: anyllm.Message{
				Role:      anyllm.RoleAssistant,
				ToolCalls: calls,
			},
			FinishReason: anyllm.FinishReasonToolCalls,
		}},
		Usage: &anyllm.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
	}
}

// ToolCallResponse returns a *ChatCompletion scripted to call one tool.
func ToolCallResponse(toolName, toolCallID string, args json.RawMessage) *anyllm.ChatCompletion {
	return &anyllm.ChatCompletion{
		Choices: []anyllm.Choice{{
			Message: anyllm.Message{
				Role: anyllm.RoleAssistant,
				ToolCalls: []anyllm.ToolCall{{
					ID: toolCallID,
					Function: anyllm.FunctionCall{
						Name:      toolName,
						Arguments: string(args),
					},
				}},
			},
			FinishReason: anyllm.FinishReasonToolCalls,
		}},
		Usage: &anyllm.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
	}
}

// TextResponse returns a *ChatCompletion containing only assistant text and no tool calls.
// Covers the most common scripted scenario in harness tests.
func TextResponse(text string) *anyllm.ChatCompletion {
	return &anyllm.ChatCompletion{
		Choices: []anyllm.Choice{{
			Message:      anyllm.Message{Role: anyllm.RoleAssistant, Content: text},
			FinishReason: anyllm.FinishReasonStop,
		}},
		Usage: &anyllm.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
	}
}
