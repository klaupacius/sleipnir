package sleipnir_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	sleipnir "sleipnir.dev/sleipnir"
	"sleipnir.dev/sleipnir/sleipnirtest"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

func mustNewHarness(t *testing.T, cfg sleipnir.Config) *sleipnir.Harness {
	t.Helper()
	h, err := sleipnir.NewHarness(cfg)
	if err != nil {
		t.Fatalf("NewHarness: %v", err)
	}
	return h
}

func defaultRouter(provider anyllm.Provider) sleipnir.MapRouter {
	return sleipnir.MapRouter{Default: sleipnir.ModelConfig{Provider: provider, Model: "stub"}}
}

// Valid Config{} -> no error
func TestNewHarness(t *testing.T) {
	h, err := sleipnir.NewHarness(sleipnir.Config{})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if h == nil {
		t.Fatal("expected non-nil Harness")
	}
}

// Bad CompactThreshold -> error
func TestNewHarnessInvalidConfig(t *testing.T) {
	_, err := sleipnir.NewHarness(sleipnir.Config{CompactThreshold: 1.5})
	if err == nil {
		t.Fatal("expected error for CompactThreshold > 1.0, got nil")
	}
}

// Register valid spec -> subsequent Run finds it
func TestRegisterAgent(t *testing.T) {
	h := mustNewHarness(t, sleipnir.Config{})
	stub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("hello"))

	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	out, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "agent",
		Prompt:    "hi",
		Router:    defaultRouter(stub),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.Text != "hello" {
		t.Errorf("expected Text %q, got %q", "hello", out.Text)
	}
}

// Spec with duplicate tool names -> ErrToolNameCollision
func TestRegisterAgentToolCollision(t *testing.T) {
	h := mustNewHarness(t, sleipnir.Config{})
	err := h.RegisterAgent(sleipnir.AgentSpec{
		Name:  "agent",
		Tools: []sleipnir.Tool{sleipnirtest.StaticTool("dup", ""), sleipnirtest.StaticTool("dup", "")},
	})
	if !errors.Is(err, sleipnir.ErrToolNameCollision) {
		t.Errorf("expected ErrToolNameCollision, got %v", err)
	}
}

// Unknown agent name -> ErrAgentNotRegistered
func TestRunUnknownAgent(t *testing.T) {
	h := mustNewHarness(t, sleipnir.Config{})
	stub := sleipnirtest.NewStubProvider(t)
	_, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "missing",
		Router:    defaultRouter(stub),
	})
	if !errors.Is(err, sleipnir.ErrAgentNotRegistered) {
		t.Errorf("expected ErrAgentNotRegistered, got %v", err)
	}
}

// Stub provider returns one text response -> RunOutput.Text correct, Stopped == StopDone
func TestRunSingleTurn(t *testing.T) {
	h := mustNewHarness(t, sleipnir.Config{})
	stub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("world"))
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	out, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "agent",
		Prompt:    "hello",
		Router:    defaultRouter(stub),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.Text != "world" {
		t.Errorf("expected Text %q, got %q", "world", out.Text)
	}
	if out.Stopped != sleipnir.StopDone {
		t.Errorf("expected Stopped %q, got %q", sleipnir.StopDone, out.Stopped)
	}
}

// Two sequential runs passing history forward -> combined history has correct message count
func TestRunMultiTurn(t *testing.T) {
	h := mustNewHarness(t, sleipnir.Config{AllowLateRegistration: true})
	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.TextResponse("response1"),
		sleipnirtest.TextResponse("response2"),
	)
	router := defaultRouter(stub)
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	out1, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "agent",
		Prompt:    "turn1",
		Router:    router,
	})
	if err != nil {
		t.Fatalf("Run turn1: %v", err)
	}

	out2, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "agent",
		History:   out1.Messages,
		Prompt:    "turn2",
		Router:    router,
	})
	if err != nil {
		t.Fatalf("Run turn2: %v", err)
	}

	// turn1-user + turn1-asst + turn2-user + turn2-asst = 4
	if len(out2.Messages) != 4 {
		t.Errorf("expected 4 messages, got %d", len(out2.Messages))
	}
}

// EventCollector receives AgentStartEvent then AgentEndEvent with StopDone
func TestRunEmitsEvents(t *testing.T) {
	h := mustNewHarness(t, sleipnir.Config{})
	stub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("ok"))
	collector := sleipnirtest.NewEventCollector()
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	if _, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "agent",
		Prompt:    "hi",
		Router:    defaultRouter(stub),
		Events:    collector,
	}); err != nil {
		t.Fatalf("Run: %v", err)
	}

	starts := sleipnirtest.ByType[sleipnir.AgentStartEvent](collector)
	ends := sleipnirtest.ByType[sleipnir.AgentEndEvent](collector)

	if len(starts) != 1 {
		t.Errorf("expected 1 AgentStartEvent, got %d", len(starts))
	} else if starts[0].AgentName != "agent" {
		t.Errorf("AgentStartEvent.AgentName = %q, want %q", starts[0].AgentName, "agent")
	}

	if len(ends) != 1 {
		t.Errorf("expected 1 AgentEndEvent, got %d", len(ends))
	} else if ends[0].Stopped != sleipnir.StopDone {
		t.Errorf("AgentEndEvent.Stopped = %q, want %q", sleipnir.StopDone, ends[0].Stopped)
	}

	if err := collector.CheckCompleted(); err != nil {
		t.Error(err)
	}
}

// RunInput.Events == nil -> run completes without panic
func TestRunNilSink(t *testing.T) {
	h := mustNewHarness(t, sleipnir.Config{})
	stub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("ok"))
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	_, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "agent",
		Prompt:    "hi",
		Router:    defaultRouter(stub),
		Events:    nil,
	})
	if err != nil {
		t.Fatalf("Run with nil sink: %v", err)
	}
}

// Register after first Run -> ErrHarnessFrozen
func TestHarnessFreeze(t *testing.T) {
	h := mustNewHarness(t, sleipnir.Config{})
	stub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("ok"))
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	if _, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "agent",
		Prompt:    "hi",
		Router:    defaultRouter(stub),
	}); err != nil {
		t.Fatalf("Run: %v", err)
	}

	err := h.RegisterAgent(sleipnir.AgentSpec{Name: "other"})
	if !errors.Is(err, sleipnir.ErrHarnessFrozen) {
		t.Errorf("expected ErrHarnessFrozen after Run, got %v", err)
	}
}

// AllowLateRegistration: true, register after Run -> succeeds
func TestHarnessAllowLateRegistration(t *testing.T) {
	h := mustNewHarness(t, sleipnir.Config{AllowLateRegistration: true})
	stub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("ok"))
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent1", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	if _, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "agent1",
		Prompt:    "hi",
		Router:    defaultRouter(stub),
	}); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent2"}); err != nil {
		t.Errorf("expected nil with AllowLateRegistration, got %v", err)
	}
}

// --- Chunk 8: tool dispatch tests ---

func stubTool(name string, result sleipnir.ToolResult) *sleipnirtest.StubTool {
	return sleipnirtest.NewStubTool(name, "", nil, func(_ context.Context, _ json.RawMessage) (sleipnir.ToolResult, error) {
		return result, nil
	})
}

// Provider returns ToolCallResponse then TextResponse -> tool invoked once, final text correct, StopDone
func TestRunSingleToolCall(t *testing.T) {
	tool := stubTool("search", sleipnir.ToolResult{Content: "result"})
	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("search", "call-1", []byte(`{"q":"hi"}`)),
		sleipnirtest.TextResponse("done"),
	)
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "a", MaxIterations: 5, Tools: []sleipnir.Tool{tool}}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	out, err := h.Run(context.Background(), sleipnir.RunInput{AgentName: "a", Prompt: "hi", Router: defaultRouter(stub)})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.Text != "done" {
		t.Errorf("expected Text %q, got %q", "done", out.Text)
	}
	if out.Stopped != sleipnir.StopDone {
		t.Errorf("expected StopDone, got %q", out.Stopped)
	}
	if tool.InvokeCount() != 1 {
		t.Errorf("expected 1 invocation, got %d", tool.InvokeCount())
	}
}

// Provider returns a response with two tool calls -> both tools invoked sequentially
func TestRunMultiToolCall(t *testing.T) {
	toolA := stubTool("toolA", sleipnir.ToolResult{Content: "a"})
	toolB := stubTool("toolB", sleipnir.ToolResult{Content: "b"})

	twoCallResp := &anyllm.ChatCompletion{
		Choices: []anyllm.Choice{{
			Message: anyllm.Message{
				Role: anyllm.RoleAssistant,
				ToolCalls: []anyllm.ToolCall{
					{ID: "c1", Function: anyllm.FunctionCall{Name: "toolA", Arguments: "{}"}},
					{ID: "c2", Function: anyllm.FunctionCall{Name: "toolB", Arguments: "{}"}},
				},
			},
			FinishReason: anyllm.FinishReasonToolCalls,
		}},
		Usage: &anyllm.Usage{PromptTokens: 10, CompletionTokens: 5},
	}

	stub := sleipnirtest.NewStubProvider(t, twoCallResp, sleipnirtest.TextResponse("done"))
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name: "a", MaxIterations: 5, Tools: []sleipnir.Tool{toolA, toolB},
	}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	if _, err := h.Run(context.Background(), sleipnir.RunInput{AgentName: "a", Prompt: "hi", Router: defaultRouter(stub)}); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if toolA.InvokeCount() != 1 {
		t.Errorf("toolA: expected 1 invocation, got %d", toolA.InvokeCount())
	}
	if toolB.InvokeCount() != 1 {
		t.Errorf("toolB: expected 1 invocation, got %d", toolB.InvokeCount())
	}
}

// Tool returns ToolResult{IsError: true} -> ToolResultEvent.IsError true; loop continues; no ErrorEvent
func TestRunToolIsError(t *testing.T) {
	tool := sleipnirtest.NewStubTool("t", "", nil, func(_ context.Context, _ json.RawMessage) (sleipnir.ToolResult, error) {
		return sleipnir.ToolResult{IsError: true, Content: "structured failure"}, nil
	})
	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("t", "c1", []byte(`{}`)),
		sleipnirtest.TextResponse("recovered"),
	)
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "a", MaxIterations: 5, Tools: []sleipnir.Tool{tool}}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}
	collector := sleipnirtest.NewEventCollector()

	out, err := h.Run(context.Background(), sleipnir.RunInput{AgentName: "a", Router: defaultRouter(stub), Events: collector})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.Text != "recovered" {
		t.Errorf("expected Text %q, got %q", "recovered", out.Text)
	}

	resultEvents := sleipnirtest.ByType[sleipnir.ToolResultEvent](collector)
	if len(resultEvents) != 1 || !resultEvents[0].IsError {
		t.Errorf("expected one ToolResultEvent with IsError=true, got %v", resultEvents)
	}
	if errs := sleipnirtest.ByType[sleipnir.ErrorEvent](collector); len(errs) != 0 {
		t.Errorf("expected no ErrorEvent for structured failure, got %d", len(errs))
	}
}

// Tool fn returns (ToolResult{}, err) -> harness wraps as IsError=true; ErrorEvent emitted; loop continues
func TestRunToolInfraError(t *testing.T) {
	infraErr := errors.New("disk full")
	tool := sleipnirtest.NewStubTool("t", "", nil, func(_ context.Context, _ json.RawMessage) (sleipnir.ToolResult, error) {
		return sleipnir.ToolResult{}, infraErr
	})
	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("t", "c1", []byte(`{}`)),
		sleipnirtest.TextResponse("ok"),
	)
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "a", MaxIterations: 5, Tools: []sleipnir.Tool{tool}}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}
	collector := sleipnirtest.NewEventCollector()

	out, err := h.Run(context.Background(), sleipnir.RunInput{AgentName: "a", Router: defaultRouter(stub), Events: collector})
	if err != nil {
		t.Fatalf("Run should continue after infra error, got %v", err)
	}
	if out.Text != "ok" {
		t.Errorf("expected Text %q, got %q", "ok", out.Text)
	}

	resultEvents := sleipnirtest.ByType[sleipnir.ToolResultEvent](collector)
	if len(resultEvents) != 1 || !resultEvents[0].IsError {
		t.Errorf("expected one ToolResultEvent with IsError=true")
	}
	errEvents := sleipnirtest.ByType[sleipnir.ErrorEvent](collector)
	if len(errEvents) != 1 || !errors.Is(errEvents[0].Err, infraErr) {
		t.Errorf("expected one ErrorEvent wrapping infraErr, got %v", errEvents)
	}
}

// Provider calls unknown tool name -> ToolResultEvent{IsError: true} with "unknown tool:" prefix; no panic
func TestRunUnknownTool(t *testing.T) {
	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("ghost", "c1", []byte(`{}`)),
		sleipnirtest.TextResponse("ok"),
	)
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "a", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}
	collector := sleipnirtest.NewEventCollector()

	if _, err := h.Run(context.Background(), sleipnir.RunInput{AgentName: "a", Router: defaultRouter(stub), Events: collector}); err != nil {
		t.Fatalf("Run: %v", err)
	}

	resultEvents := sleipnirtest.ByType[sleipnir.ToolResultEvent](collector)
	if len(resultEvents) != 1 {
		t.Fatalf("expected 1 ToolResultEvent, got %d", len(resultEvents))
	}
	if !resultEvents[0].IsError {
		t.Error("expected IsError=true for unknown tool")
	}
	if resultEvents[0].Result == "" {
		t.Error("expected non-empty content with 'unknown tool:' prefix")
	}
}

// EventCollector.ToolCalls() and ByType[ToolResultEvent] return correct counts
func TestRunToolCallEvents(t *testing.T) {
	tool := stubTool("search", sleipnir.ToolResult{Content: "r"})
	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("search", "c1", []byte(`{}`)),
		sleipnirtest.TextResponse("done"),
	)
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "a", MaxIterations: 5, Tools: []sleipnir.Tool{tool}}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}
	collector := sleipnirtest.NewEventCollector()

	if _, err := h.Run(context.Background(), sleipnir.RunInput{AgentName: "a", Router: defaultRouter(stub), Events: collector}); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if calls := collector.ToolCalls(); len(calls) != 1 {
		t.Errorf("expected 1 ToolCallEvent, got %d", len(calls))
	}
	if results := sleipnirtest.ByType[sleipnir.ToolResultEvent](collector); len(results) != 1 {
		t.Errorf("expected 1 ToolResultEvent, got %d", len(results))
	}
}

// TextResponse -> TokenEvent emitted with correct text
func TestRunTokenEvent(t *testing.T) {
	stub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("hello world"))
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "a", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}
	collector := sleipnirtest.NewEventCollector()

	if _, err := h.Run(context.Background(), sleipnir.RunInput{AgentName: "a", Prompt: "hi", Router: defaultRouter(stub), Events: collector}); err != nil {
		t.Fatalf("Run: %v", err)
	}

	tokens := sleipnirtest.ByType[sleipnir.TokenEvent](collector)
	if len(tokens) != 1 {
		t.Fatalf("expected 1 TokenEvent, got %d", len(tokens))
	}
	if tokens[0].Text != "hello world" {
		t.Errorf("expected Text %q, got %q", "hello world", tokens[0].Text)
	}
}

// Response with Reasoning content -> ThinkingEvent emitted
func TestRunThinkingEvent(t *testing.T) {
	thinkingResp := &anyllm.ChatCompletion{
		Choices: []anyllm.Choice{{
			Message: anyllm.Message{
				Role:      anyllm.RoleAssistant,
				Content:   "answer",
				Reasoning: &anyllm.Reasoning{Content: "let me think"},
			},
			FinishReason: anyllm.FinishReasonStop,
		}},
		Usage: &anyllm.Usage{PromptTokens: 10, CompletionTokens: 5},
	}
	stub := sleipnirtest.NewStubProvider(t, thinkingResp)
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "a", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}
	collector := sleipnirtest.NewEventCollector()

	if _, err := h.Run(context.Background(), sleipnir.RunInput{AgentName: "a", Prompt: "hi", Router: defaultRouter(stub), Events: collector}); err != nil {
		t.Fatalf("Run: %v", err)
	}

	thinking := sleipnirtest.ByType[sleipnir.ThinkingEvent](collector)
	if len(thinking) != 1 {
		t.Fatalf("expected 1 ThinkingEvent, got %d", len(thinking))
	}
	if thinking[0].Text != "let me think" {
		t.Errorf("expected Text %q, got %q", "let me think", thinking[0].Text)
	}
}
