package sleipnir_test

import (
	"context"
	"encoding/json"
	"errors"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

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

// Spec with duplicate tool names -> error message contains the colliding tool name
func TestRegisterAgentToolCollisionIncludesToolName(t *testing.T) {
	h := mustNewHarness(t, sleipnir.Config{})
	const toolName = "my_special_xyz_tool"
	err := h.RegisterAgent(sleipnir.AgentSpec{
		Name:  "agent",
		Tools: []sleipnir.Tool{sleipnirtest.StaticTool(toolName, ""), sleipnirtest.StaticTool(toolName, "")},
	})
	if err == nil {
		t.Fatal("expected error for duplicate tool, got nil")
	}
	if !errors.Is(err, sleipnir.ErrToolNameCollision) {
		t.Fatalf("expected ErrToolNameCollision, got %v", err)
	}
	if !strings.Contains(err.Error(), toolName) {
		t.Errorf("error message should contain the duplicate tool name %q, got: %q", toolName, err.Error())
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

// --- Chunk 9: parallel dispatch and ExtraTools tests ---

// MultiToolCallResponse with two distinct tools -> both invoked, results in call-index order, StopDone
func TestRunParallelTwoToolCalls(t *testing.T) {
	toolA := stubTool("toolA", sleipnir.ToolResult{Content: "a"})
	toolB := stubTool("toolB", sleipnir.ToolResult{Content: "b"})

	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.MultiToolCallResponse(
			anyllm.ToolCall{ID: "c1", Function: anyllm.FunctionCall{Name: "toolA", Arguments: "{}"}},
			anyllm.ToolCall{ID: "c2", Function: anyllm.FunctionCall{Name: "toolB", Arguments: "{}"}},
		),
		sleipnirtest.TextResponse("done"),
	)
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "a", MaxIterations: 5, Tools: []sleipnir.Tool{toolA, toolB}}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	out, err := h.Run(context.Background(), sleipnir.RunInput{AgentName: "a", Router: defaultRouter(stub)})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.Stopped != sleipnir.StopDone {
		t.Errorf("expected StopDone, got %q", out.Stopped)
	}
	if toolA.InvokeCount() != 1 {
		t.Errorf("toolA: expected 1 invocation, got %d", toolA.InvokeCount())
	}
	if toolB.InvokeCount() != 1 {
		t.Errorf("toolB: expected 1 invocation, got %d", toolB.InvokeCount())
	}
}

// MaxParallelTools: 2, 4 tools -> at most 2 concurrently inside Invoke at any instant
func TestRunParallelSemaphoreBound(t *testing.T) {
	const numTools = 4
	const maxParallel = 2

	var activeMu sync.Mutex
	var active int
	var exceeded bool

	makeBlockingTool := func(name string, ready *sync.WaitGroup, release chan struct{}) *sleipnirtest.StubTool {
		return sleipnirtest.NewStubTool(name, "", nil, func(_ context.Context, _ json.RawMessage) (sleipnir.ToolResult, error) {
			activeMu.Lock()
			active++
			if active > maxParallel {
				exceeded = true
			}
			activeMu.Unlock()

			if ready != nil {
				ready.Done()
			}
			if release != nil {
				<-release
			}

			activeMu.Lock()
			active--
			activeMu.Unlock()
			return sleipnir.ToolResult{Content: name}, nil
		})
	}

	// Two slow tools that hold the semaphore until signalled, two fast tools.
	ready := &sync.WaitGroup{}
	ready.Add(maxParallel)
	release := make(chan struct{})

	tools := make([]sleipnir.Tool, numTools)
	calls := make([]anyllm.ToolCall, numTools)
	names := []string{"t0", "t1", "t2", "t3"}
	for i := range numTools {
		var r *sync.WaitGroup
		var rel chan struct{}
		if i < maxParallel {
			r, rel = ready, release
		}
		tools[i] = makeBlockingTool(names[i], r, rel)
		calls[i] = anyllm.ToolCall{ID: names[i], Function: anyllm.FunctionCall{Name: names[i], Arguments: "{}"}}
	}

	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.MultiToolCallResponse(calls...),
		sleipnirtest.TextResponse("done"),
	)
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name: "a", MaxIterations: 5, MaxParallelTools: maxParallel, Tools: tools,
	}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		h.Run(context.Background(), sleipnir.RunInput{AgentName: "a", Router: defaultRouter(stub)}) //nolint
	}()

	// Wait until the first maxParallel tools are inside Invoke, then release them.
	ready.Wait()
	close(release)
	<-done

	if exceeded {
		t.Errorf("semaphore exceeded: more than %d tools ran concurrently", maxParallel)
	}
}

// Cancel ctx while a tool holds the semaphore -> run terminates with context.Canceled, no leak
func TestRunParallelContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	t1Started := make(chan struct{})
	t1Release := make(chan struct{})

	// tool1 holds the semaphore until we release it; we cancel ctx while it waits.
	tool1 := sleipnirtest.NewStubTool("t1", "", nil, func(_ context.Context, _ json.RawMessage) (sleipnir.ToolResult, error) {
		close(t1Started)
		<-t1Release
		return sleipnir.ToolResult{Content: "t1"}, nil
	})
	tool2 := stubTool("t2", sleipnir.ToolResult{Content: "t2"})

	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.MultiToolCallResponse(
			anyllm.ToolCall{ID: "c1", Function: anyllm.FunctionCall{Name: "t1", Arguments: "{}"}},
			anyllm.ToolCall{ID: "c2", Function: anyllm.FunctionCall{Name: "t2", Arguments: "{}"}},
		),
		// No second response: after tools finish, callProvider checks ctx.Err() → context.Canceled.
	)
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name: "a", MaxIterations: 5, MaxParallelTools: 1, Tools: []sleipnir.Tool{tool1, tool2},
	}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	runDone := make(chan error, 1)
	go func() {
		_, err := h.Run(ctx, sleipnir.RunInput{AgentName: "a", Router: defaultRouter(stub)})
		runDone <- err
	}()

	<-t1Started      // t1 is running, holding the semaphore; t2 is blocked on semaphore acquire
	cancel()         // cancel ctx while t2 waits
	close(t1Release) // let t1 finish and release the semaphore

	err := <-runDone
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}

// Tool in RunInput.ExtraTools only -> provider can call it, result returned correctly
func TestRunExtraToolsMerged(t *testing.T) {
	extra := stubTool("extra", sleipnir.ToolResult{Content: "extra result"})
	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("extra", "c1", []byte(`{}`)),
		sleipnirtest.TextResponse("done"),
	)
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "a", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	out, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName:  "a",
		Router:     defaultRouter(stub),
		ExtraTools: []sleipnir.Tool{extra},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if extra.InvokeCount() != 1 {
		t.Errorf("expected extra tool invoked once, got %d", extra.InvokeCount())
	}
	if out.Text != "done" {
		t.Errorf("expected Text %q, got %q", "done", out.Text)
	}
}

// ExtraTools name collision with spec.Tools -> ErrToolNameCollision before any LLM call
func TestRunExtraToolsCollision(t *testing.T) {
	specTool := stubTool("dup", sleipnir.ToolResult{Content: "spec"})
	extraTool := stubTool("dup", sleipnir.ToolResult{Content: "extra"})
	stub := sleipnirtest.NewStubProvider(t) // no responses — should not be called
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "a", MaxIterations: 5, Tools: []sleipnir.Tool{specTool}}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	_, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName:  "a",
		Router:     defaultRouter(stub),
		ExtraTools: []sleipnir.Tool{extraTool},
	})
	if !errors.Is(err, sleipnir.ErrToolNameCollision) {
		t.Errorf("expected ErrToolNameCollision, got %v", err)
	}
}

// RunInput.ExtraTools nil -> run proceeds without panic
func TestRunExtraToolsNil(t *testing.T) {
	stub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("ok"))
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "a", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}
	_, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName:  "a",
		Router:     defaultRouter(stub),
		ExtraTools: nil,
	})
	if err != nil {
		t.Fatalf("Run with nil ExtraTools: %v", err)
	}
}

// 3 tools sharing a counter via sync.Mutex, MaxParallelTools: 3, run under -race -> no data race
func TestRunParallelRace(t *testing.T) {
	var mu sync.Mutex
	var counter int

	makeCountingTool := func(name string) *sleipnirtest.StubTool {
		return sleipnirtest.NewStubTool(name, "", nil, func(_ context.Context, _ json.RawMessage) (sleipnir.ToolResult, error) {
			mu.Lock()
			counter++
			mu.Unlock()
			return sleipnir.ToolResult{Content: name}, nil
		})
	}

	tools := []sleipnir.Tool{makeCountingTool("r0"), makeCountingTool("r1"), makeCountingTool("r2")}
	calls := []anyllm.ToolCall{
		{ID: "r0", Function: anyllm.FunctionCall{Name: "r0", Arguments: "{}"}},
		{ID: "r1", Function: anyllm.FunctionCall{Name: "r1", Arguments: "{}"}},
		{ID: "r2", Function: anyllm.FunctionCall{Name: "r2", Arguments: "{}"}},
	}
	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.MultiToolCallResponse(calls...),
		sleipnirtest.TextResponse("done"),
	)
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name: "a", MaxIterations: 5, MaxParallelTools: 3, Tools: tools,
	}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	if _, err := h.Run(context.Background(), sleipnir.RunInput{AgentName: "a", Router: defaultRouter(stub)}); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if counter != 3 {
		t.Errorf("expected counter=3, got %d", counter)
	}
}

// --- Chunk 10: sub-agent tests ---

// CaptureProvider wraps StubProvider and captures every CompletionParams it is called with.
type CaptureProvider struct {
	*sleipnirtest.StubProvider
	mu     sync.Mutex
	params []anyllm.CompletionParams
}

func newCaptureProvider(t *testing.T, responses ...*anyllm.ChatCompletion) *CaptureProvider {
	return &CaptureProvider{StubProvider: sleipnirtest.NewStubProvider(t, responses...)}
}

func (c *CaptureProvider) Completion(ctx context.Context, p anyllm.CompletionParams) (*anyllm.ChatCompletion, error) {
	c.mu.Lock()
	c.params = append(c.params, p)
	c.mu.Unlock()
	return c.StubProvider.Completion(ctx, p)
}

func (c *CaptureProvider) CapturedParams() []anyllm.CompletionParams {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]anyllm.CompletionParams, len(c.params))
	copy(out, c.params)
	return out
}

// AgentAsTool on a registered agent -> no error; Definition matches spec
func TestAgentAsToolRegistered(t *testing.T) {
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name:        "child",
		Description: "child agent",
		MaxIterations: 5,
	}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	tool, err := h.AgentAsTool("child")
	if err != nil {
		t.Fatalf("AgentAsTool: %v", err)
	}
	if tool.Definition().Name != "child" {
		t.Errorf("Definition().Name = %q, want %q", tool.Definition().Name, "child")
	}
	if tool.Definition().Description != "child agent" {
		t.Errorf("Definition().Description = %q, want %q", tool.Definition().Description, "child agent")
	}
}

// AgentAsTool on unknown name -> ErrAgentNotRegistered
func TestAgentAsToolNotRegistered(t *testing.T) {
	h := mustNewHarness(t, sleipnir.Config{})

	_, err := h.AgentAsTool("unknown")
	if !errors.Is(err, sleipnir.ErrAgentNotRegistered) {
		t.Errorf("expected ErrAgentNotRegistered, got %v", err)
	}
}

// Parent provider calls sub-agent tool; sub-agent returns text; parent receives ToolResultEvent
// and continues to StopDone.
func TestRunSubAgentBasic(t *testing.T) {
	childStub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("sub result"))
	parentStub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("child", "tc-1", []byte(`{"input":"hi"}`)),
		sleipnirtest.TextResponse("parent done"),
	)

	h := mustNewHarness(t, sleipnir.Config{AllowLateRegistration: true})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "child", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent child: %v", err)
	}
	childTool, err := h.AgentAsTool("child")
	if err != nil {
		t.Fatalf("AgentAsTool: %v", err)
	}
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name:          "parent",
		MaxIterations: 5,
		Tools:         []sleipnir.Tool{childTool},
	}); err != nil {
		t.Fatalf("RegisterAgent parent: %v", err)
	}

	router := sleipnir.MapRouter{
		Overrides: map[string]sleipnir.ModelConfig{
			"child":  {Provider: childStub, Model: "stub"},
			"parent": {Provider: parentStub, Model: "stub"},
		},
	}
	collector := sleipnirtest.NewEventCollector()

	out, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "parent",
		Prompt:    "go",
		Router:    router,
		Events:    collector,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.Stopped != sleipnir.StopDone {
		t.Errorf("expected StopDone, got %q", out.Stopped)
	}

	// The ToolResultEvent for the sub-agent call should have the sub-agent text.
	results := sleipnirtest.ByType[sleipnir.ToolResultEvent](collector)
	var parentResults []sleipnir.ToolResultEvent
	for _, r := range results {
		if r.AgentName == "parent" {
			parentResults = append(parentResults, r)
		}
	}
	if len(parentResults) != 1 {
		t.Fatalf("expected 1 parent ToolResultEvent, got %d", len(parentResults))
	}
	if parentResults[0].Result != "sub result" {
		t.Errorf("ToolResultEvent.Result = %q, want %q", parentResults[0].Result, "sub result")
	}
	if parentResults[0].IsError {
		t.Error("expected IsError=false for successful sub-agent")
	}
}

// Sub-agent AgentStartEvent.ParentName == parent name; sub-agent AgentEndEvent appears before
// parent's ToolResultEvent for that call.
func TestRunSubAgentEvents(t *testing.T) {
	childStub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("child text"))
	parentStub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("child", "tc-1", []byte(`{}`)),
		sleipnirtest.TextResponse("parent done"),
	)

	h := mustNewHarness(t, sleipnir.Config{AllowLateRegistration: true})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "child", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent child: %v", err)
	}
	childTool, _ := h.AgentAsTool("child")
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name: "parent", MaxIterations: 5, Tools: []sleipnir.Tool{childTool},
	}); err != nil {
		t.Fatalf("RegisterAgent parent: %v", err)
	}

	router := sleipnir.MapRouter{
		Overrides: map[string]sleipnir.ModelConfig{
			"child":  {Provider: childStub, Model: "stub"},
			"parent": {Provider: parentStub, Model: "stub"},
		},
	}
	collector := sleipnirtest.NewEventCollector()

	if _, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "parent", Router: router, Events: collector,
	}); err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Verify sub-agent AgentStartEvent has ParentName == "parent"
	starts := sleipnirtest.ByType[sleipnir.AgentStartEvent](collector)
	var childStart *sleipnir.AgentStartEvent
	for i := range starts {
		if starts[i].AgentName == "child" {
			childStart = &starts[i]
			break
		}
	}
	if childStart == nil {
		t.Fatal("no AgentStartEvent for child")
	}
	if childStart.ParentName != "parent" {
		t.Errorf("child AgentStartEvent.ParentName = %q, want %q", childStart.ParentName, "parent")
	}

	// Verify ordering: child AgentEndEvent appears before parent ToolResultEvent
	allEvents := collector.Events()
	childEndIdx := -1
	parentResultIdx := -1
	for i, e := range allEvents {
		if end, ok := e.(sleipnir.AgentEndEvent); ok && end.AgentName == "child" {
			childEndIdx = i
		}
		if res, ok := e.(sleipnir.ToolResultEvent); ok && res.AgentName == "parent" {
			parentResultIdx = i
		}
	}
	if childEndIdx == -1 {
		t.Fatal("no child AgentEndEvent found")
	}
	if parentResultIdx == -1 {
		t.Fatal("no parent ToolResultEvent found")
	}
	if childEndIdx >= parentResultIdx {
		t.Errorf("expected child AgentEndEvent (idx %d) before parent ToolResultEvent (idx %d)", childEndIdx, parentResultIdx)
	}
}

// Sub-agent's provider receives only its own messages — not the parent's history.
func TestRunSubAgentIsolatedHistory(t *testing.T) {
	childCapture := newCaptureProvider(t, sleipnirtest.TextResponse("child result"))
	parentStub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("child", "tc-1", []byte(`{"q":"hello"}`)),
		sleipnirtest.TextResponse("parent done"),
	)

	h := mustNewHarness(t, sleipnir.Config{AllowLateRegistration: true})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "child", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent child: %v", err)
	}
	childTool, _ := h.AgentAsTool("child")
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name: "parent", MaxIterations: 5, Tools: []sleipnir.Tool{childTool},
	}); err != nil {
		t.Fatalf("RegisterAgent parent: %v", err)
	}

	router := sleipnir.MapRouter{
		Overrides: map[string]sleipnir.ModelConfig{
			"child":  {Provider: childCapture, Model: "stub"},
			"parent": {Provider: parentStub, Model: "stub"},
		},
	}

	if _, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "parent",
		Prompt:    "parent prompt that should not appear in child history",
		Router:    router,
	}); err != nil {
		t.Fatalf("Run: %v", err)
	}

	captured := childCapture.CapturedParams()
	if len(captured) != 1 {
		t.Fatalf("expected child provider called once, got %d", len(captured))
	}
	for _, msg := range captured[0].Messages {
		if msg.Role == anyllm.RoleUser {
			if s, ok := msg.Content.(string); ok && strings.Contains(s, "parent prompt") {
				t.Error("child received parent's prompt in its history")
			}
		}
	}
}

// ExtraTools inherited by sub-agent (OmitExtraToolsInheritance: false).
func TestRunSubAgentExtraToolsInherited(t *testing.T) {
	extraInvoked := make(chan struct{}, 1)
	extraTool := sleipnirtest.NewStubTool("extra", "", nil, func(_ context.Context, _ json.RawMessage) (sleipnir.ToolResult, error) {
		select {
		case extraInvoked <- struct{}{}:
		default:
		}
		return sleipnir.ToolResult{Content: "extra result"}, nil
	})

	// child: calls extra tool, then returns text
	childStub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("extra", "ec-1", []byte(`{}`)),
		sleipnirtest.TextResponse("child done"),
	)
	// parent: calls child sub-agent, then returns text
	parentStub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("child", "tc-1", []byte(`{}`)),
		sleipnirtest.TextResponse("parent done"),
	)

	h := mustNewHarness(t, sleipnir.Config{AllowLateRegistration: true})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "child", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent child: %v", err)
	}
	childTool, _ := h.AgentAsTool("child")
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name: "parent", MaxIterations: 5, Tools: []sleipnir.Tool{childTool},
	}); err != nil {
		t.Fatalf("RegisterAgent parent: %v", err)
	}

	router := sleipnir.MapRouter{
		Overrides: map[string]sleipnir.ModelConfig{
			"child":  {Provider: childStub, Model: "stub"},
			"parent": {Provider: parentStub, Model: "stub"},
		},
	}

	if _, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName:                 "parent",
		Router:                    router,
		ExtraTools:                []sleipnir.Tool{extraTool},
		OmitExtraToolsInheritance: false,
	}); err != nil {
		t.Fatalf("Run: %v", err)
	}

	select {
	case <-extraInvoked:
		// good
	default:
		t.Error("expected extra tool to be invoked by child, but it was not")
	}
}

// ExtraTools NOT inherited when OmitExtraToolsInheritance: true.
func TestRunSubAgentExtraToolsNotInherited(t *testing.T) {
	extraTool := sleipnirtest.NewStubTool("extra", "", nil, func(_ context.Context, _ json.RawMessage) (sleipnir.ToolResult, error) {
		return sleipnir.ToolResult{Content: "extra result"}, nil
	})

	// child: tries to call extra tool (unknown), then returns text
	childStub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("extra", "ec-1", []byte(`{}`)),
		sleipnirtest.TextResponse("child done"),
	)
	parentStub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("child", "tc-1", []byte(`{}`)),
		sleipnirtest.TextResponse("parent done"),
	)

	h := mustNewHarness(t, sleipnir.Config{AllowLateRegistration: true})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "child", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent child: %v", err)
	}
	childTool, _ := h.AgentAsTool("child")
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name: "parent", MaxIterations: 5, Tools: []sleipnir.Tool{childTool},
	}); err != nil {
		t.Fatalf("RegisterAgent parent: %v", err)
	}

	router := sleipnir.MapRouter{
		Overrides: map[string]sleipnir.ModelConfig{
			"child":  {Provider: childStub, Model: "stub"},
			"parent": {Provider: parentStub, Model: "stub"},
		},
	}
	collector := sleipnirtest.NewEventCollector()

	out, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName:                 "parent",
		Router:                    router,
		ExtraTools:                []sleipnir.Tool{extraTool},
		OmitExtraToolsInheritance: true,
		Events:                    collector,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.Stopped != sleipnir.StopDone {
		t.Errorf("expected parent StopDone, got %q", out.Stopped)
	}

	// The child should have received an unknown-tool error for "extra"
	childResults := sleipnirtest.ByType[sleipnir.ToolResultEvent](collector)
	var childUnknown bool
	for _, r := range childResults {
		if r.AgentName == "child" && r.IsError {
			childUnknown = true
		}
	}
	if !childUnknown {
		t.Error("expected child to receive IsError=true for unknown 'extra' tool")
	}
	// extra tool should NOT have been invoked
	if extraTool.InvokeCount() != 0 {
		t.Errorf("extra tool should not be invoked, got %d", extraTool.InvokeCount())
	}
}

// Sub-agent hitting MaxIterations -> ErrIterationBudget; parent receives ToolResultEvent{IsError: true};
// parent run continues.
func TestRunSubAgentIterationBudget(t *testing.T) {
	// Child always returns a tool call (never text), MaxIterations=1 -> hits ErrIterationBudget
	// The child's "loop" tool just returns something, but the child will try to call it once
	// and then exhaust its budget.
	childLoopTool := sleipnirtest.NewStubTool("loop", "", nil, func(_ context.Context, _ json.RawMessage) (sleipnir.ToolResult, error) {
		return sleipnir.ToolResult{Content: "looping"}, nil
	})
	// child always responds with a tool call
	childStub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("loop", "lc-1", []byte(`{}`)),
		// no text response — budget exhausted after 1 iteration
	)
	parentStub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("child", "tc-1", []byte(`{}`)),
		sleipnirtest.TextResponse("parent done after child budget error"),
	)

	h := mustNewHarness(t, sleipnir.Config{AllowLateRegistration: true})
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name:          "child",
		MaxIterations: 1, // hits budget immediately after one tool call
		Tools:         []sleipnir.Tool{childLoopTool},
	}); err != nil {
		t.Fatalf("RegisterAgent child: %v", err)
	}
	childTool, _ := h.AgentAsTool("child")
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name:          "parent",
		MaxIterations: 5,
		Tools:         []sleipnir.Tool{childTool},
	}); err != nil {
		t.Fatalf("RegisterAgent parent: %v", err)
	}

	router := sleipnir.MapRouter{
		Overrides: map[string]sleipnir.ModelConfig{
			"child":  {Provider: childStub, Model: "stub"},
			"parent": {Provider: parentStub, Model: "stub"},
		},
	}
	collector := sleipnirtest.NewEventCollector()

	out, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "parent",
		Router:    router,
		Events:    collector,
	})
	if err != nil {
		t.Fatalf("parent Run should succeed, got %v", err)
	}
	if out.Stopped != sleipnir.StopDone {
		t.Errorf("expected parent StopDone, got %q", out.Stopped)
	}

	// Parent should have received an IsError ToolResultEvent for the child sub-agent call
	results := sleipnirtest.ByType[sleipnir.ToolResultEvent](collector)
	var parentChildResult *sleipnir.ToolResultEvent
	for i := range results {
		if results[i].AgentName == "parent" && results[i].ToolCallID == "tc-1" {
			parentChildResult = &results[i]
		}
	}
	if parentChildResult == nil {
		t.Fatal("no parent ToolResultEvent for child sub-agent call")
	}
	if !parentChildResult.IsError {
		t.Error("expected IsError=true for child hitting iteration budget")
	}
	if !strings.Contains(parentChildResult.Result, "iteration_budget") {
		t.Errorf("expected result to contain 'iteration_budget', got %q", parentChildResult.Result)
	}
}

// Cancel ctx while sub-agent is mid-run -> parent run terminates with context.Canceled.
func TestRunSubAgentContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	childStarted := make(chan struct{})
	childRelease := make(chan struct{})

	// A blocking tool inside the child: signals childStarted and blocks on childRelease.
	blockingTool := sleipnirtest.NewStubTool("block", "", nil, func(_ context.Context, _ json.RawMessage) (sleipnir.ToolResult, error) {
		close(childStarted)
		<-childRelease
		return sleipnir.ToolResult{Content: "unblocked"}, nil
	})

	childStub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("block", "bc-1", []byte(`{}`)),
		// no second response needed: context will be cancelled
	)
	parentStub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("child", "tc-1", []byte(`{}`)),
	)

	h := mustNewHarness(t, sleipnir.Config{AllowLateRegistration: true})
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name:          "child",
		MaxIterations: 5,
		Tools:         []sleipnir.Tool{blockingTool},
	}); err != nil {
		t.Fatalf("RegisterAgent child: %v", err)
	}
	childTool, _ := h.AgentAsTool("child")
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name:          "parent",
		MaxIterations: 5,
		Tools:         []sleipnir.Tool{childTool},
	}); err != nil {
		t.Fatalf("RegisterAgent parent: %v", err)
	}

	router := sleipnir.MapRouter{
		Overrides: map[string]sleipnir.ModelConfig{
			"child":  {Provider: childStub, Model: "stub"},
			"parent": {Provider: parentStub, Model: "stub"},
		},
	}

	runDone := make(chan error, 1)
	go func() {
		_, err := h.Run(ctx, sleipnir.RunInput{AgentName: "parent", Router: router})
		runDone <- err
	}()

	// Wait until child's blocking tool is running, then cancel context.
	<-childStarted
	cancel()
	close(childRelease) // unblock the tool so goroutines can clean up

	err := <-runDone
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}

// MaxTotalTokens: 0 -> unlimited; run completes with StopDone regardless of usage.
func TestRunTokenBudgetUnlimited(t *testing.T) {
	h := mustNewHarness(t, sleipnir.Config{})
	stub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("done"))
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	out, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName:      "agent",
		Prompt:         "hi",
		Router:         defaultRouter(stub),
		MaxTotalTokens: 0, // unlimited
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.Stopped != sleipnir.StopDone {
		t.Errorf("expected StopDone, got %q", out.Stopped)
	}
}

// Budget set above total usage -> StopDone, no error.
func TestRunTokenBudgetNotExceeded(t *testing.T) {
	h := mustNewHarness(t, sleipnir.Config{})
	// One response: TotalTokens = 15; budget = 100
	stub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("ok"))
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	out, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName:      "agent",
		Prompt:         "hi",
		Router:         defaultRouter(stub),
		MaxTotalTokens: 100,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.Stopped != sleipnir.StopDone {
		t.Errorf("expected StopDone, got %q", out.Stopped)
	}
}

// Budget set below first response's usage -> ErrTokenBudget, StopTokenBudget, partial history.
func TestRunTokenBudgetExceeded(t *testing.T) {
	h := mustNewHarness(t, sleipnir.Config{})
	// StubProvider TotalTokens = 15 per response; budget = 10 < 15
	stub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("should not appear"))
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	out, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName:      "agent",
		Prompt:         "hi",
		Router:         defaultRouter(stub),
		MaxTotalTokens: 10, // budget < 15 tokens from first response
	})
	if !errors.Is(err, sleipnir.ErrTokenBudget) {
		t.Fatalf("expected ErrTokenBudget, got %v", err)
	}
	if out.Stopped != sleipnir.StopTokenBudget {
		t.Errorf("expected StopTokenBudget, got %q", out.Stopped)
	}
	if len(out.Messages) == 0 {
		t.Error("expected non-empty Messages in partial history")
	}
}

// Budget exactly equals usage -> StopTokenBudget (budget check uses >=).
func TestRunTokenBudgetExactlyMet(t *testing.T) {
	h := mustNewHarness(t, sleipnir.Config{})
	// StubProvider TotalTokens = 15; budget = 15 exactly
	stub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("exact"))
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	out, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName:      "agent",
		Prompt:         "hi",
		Router:         defaultRouter(stub),
		MaxTotalTokens: 15, // budget == 15 tokens from first response
	})
	if !errors.Is(err, sleipnir.ErrTokenBudget) {
		t.Fatalf("expected ErrTokenBudget, got %v", err)
	}
	if out.Stopped != sleipnir.StopTokenBudget {
		t.Errorf("expected StopTokenBudget, got %q", out.Stopped)
	}
}

// Sub-agent token usage contributes to shared budget; combined usage exceeds budget.
func TestRunTokenBudgetSubAgentContributes(t *testing.T) {
	// Parent: budget = 20. Parent first response = 15 tokens (under budget).
	// Sub-agent response = 15 tokens -> shared total = 30 (> 20).
	// The child's runLoop sees overBudget() after its own response, so it returns ErrTokenBudget.
	// subAgentResultToToolResult converts that to an IsError ToolResult (not an infraErr),
	// so the parent gets an error tool result and continues. On the parent's next LLM call,
	// the shared counter (30) is already over budget (20), triggering ErrTokenBudget for the parent.
	h := mustNewHarness(t, sleipnir.Config{AllowLateRegistration: true})

	childStub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("child done"))
	parentStub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("child", "tc-1", []byte(`{}`)),
		sleipnirtest.TextResponse("parent done"),
	)

	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name:          "child",
		MaxIterations: 5,
	}); err != nil {
		t.Fatalf("RegisterAgent child: %v", err)
	}
	childTool, _ := h.AgentAsTool("child")
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name:          "parent",
		MaxIterations: 5,
		Tools:         []sleipnir.Tool{childTool},
	}); err != nil {
		t.Fatalf("RegisterAgent parent: %v", err)
	}

	router := sleipnir.MapRouter{
		Overrides: map[string]sleipnir.ModelConfig{
			"child":  {Provider: childStub, Model: "stub"},
			"parent": {Provider: parentStub, Model: "stub"},
		},
		Default: sleipnir.ModelConfig{Provider: parentStub, Model: "stub"},
	}

	// Budget = 20: parent first call (15) stays under; child call (15) pushes total to 30.
	out, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName:      "parent",
		Prompt:         "go",
		Router:         router,
		MaxTotalTokens: 20,
	})
	if !errors.Is(err, sleipnir.ErrTokenBudget) {
		t.Fatalf("expected ErrTokenBudget, got %v", err)
	}
	if out.Stopped != sleipnir.StopTokenBudget {
		t.Errorf("expected StopTokenBudget, got %q", out.Stopped)
	}
}

// Multi-turn run via separate Run calls -> each RunOutput.Usage.TotalTokens equals per-run usage.
func TestRunUsageAccumulated(t *testing.T) {
	h := mustNewHarness(t, sleipnir.Config{AllowLateRegistration: true})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	stub1 := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("turn1"))
	out1, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "agent",
		Prompt:    "hello",
		Router:    defaultRouter(stub1),
	})
	if err != nil {
		t.Fatalf("Run turn1: %v", err)
	}

	stub2 := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("turn2"))
	out2, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "agent",
		History:   out1.Messages,
		Prompt:    "continue",
		Router:    defaultRouter(stub2),
	})
	if err != nil {
		t.Fatalf("Run turn2: %v", err)
	}

	// Each Run creates an independent runState; each single-turn run reports TotalTokens = 15.
	if out1.Usage.TotalTokens != 15 {
		t.Errorf("turn1 TotalTokens = %d, want 15", out1.Usage.TotalTokens)
	}
	if out2.Usage.TotalTokens != 15 {
		t.Errorf("turn2 TotalTokens = %d, want 15", out2.Usage.TotalTokens)
	}
}

// Parent calls sub-agent; RunOutput.Usage.TotalTokens equals parent usage + sub-agent usage.
func TestRunUsageIncludesSubAgent(t *testing.T) {
	h := mustNewHarness(t, sleipnir.Config{AllowLateRegistration: true})

	// child: 1 response = 15 tokens
	childStub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("child answer"))
	// parent: tool call response (15 tokens) + final text response (15 tokens) = 30 tokens
	parentStub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("child", "tc-1", []byte(`{}`)),
		sleipnirtest.TextResponse("parent answer"),
	)

	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name:          "child",
		MaxIterations: 5,
	}); err != nil {
		t.Fatalf("RegisterAgent child: %v", err)
	}
	childTool, _ := h.AgentAsTool("child")
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name:          "parent",
		MaxIterations: 5,
		Tools:         []sleipnir.Tool{childTool},
	}); err != nil {
		t.Fatalf("RegisterAgent parent: %v", err)
	}

	router := sleipnir.MapRouter{
		Overrides: map[string]sleipnir.ModelConfig{
			"child":  {Provider: childStub, Model: "stub"},
			"parent": {Provider: parentStub, Model: "stub"},
		},
		Default: sleipnir.ModelConfig{Provider: parentStub, Model: "stub"},
	}

	out, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "parent",
		Prompt:    "go",
		Router:    router,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// parent: 2 LLM calls * 15 = 30 tokens local; child: 1 LLM call * 15 = 15 tokens sub-usage.
	// Total reported in parent RunOutput = 30 (parent local) + 15 (sub-agent) = 45.
	wantTotal := int64(45)
	if out.Usage.TotalTokens != wantTotal {
		t.Errorf("TotalTokens = %d, want %d", out.Usage.TotalTokens, wantTotal)
	}
}

// --- Chunk 15: HITL tests ---

// funcHITLHandler is a test helper that wraps a function as a HITLHandler.
type funcHITLHandler struct {
	fn func(ctx context.Context, agent, question, contextBlurb string) (string, error)
}

func (h *funcHITLHandler) AskUser(ctx context.Context, agent, question, contextBlurb string) (string, error) {
	return h.fn(ctx, agent, question, contextBlurb)
}

// TestAskUserToolInvoked: agent calls ask_user; handler returns "user reply"; run ends with StopDone.
func TestAskUserToolInvoked(t *testing.T) {
	handler := &funcHITLHandler{
		fn: func(_ context.Context, _, _, _ string) (string, error) {
			return "user reply", nil
		},
	}
	askTool := sleipnir.AskUserTool(handler)

	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("ask_user", "q-1", []byte(`{"question":"What is your name?"}`)),
		sleipnirtest.TextResponse("Thanks for the reply"),
	)
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	out, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName:  "agent",
		Prompt:     "go",
		Router:     defaultRouter(stub),
		ExtraTools: []sleipnir.Tool{askTool},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.Stopped != sleipnir.StopDone {
		t.Errorf("Stopped = %q, want StopDone", out.Stopped)
	}
	if out.Text != "Thanks for the reply" {
		t.Errorf("Text = %q, want %q", out.Text, "Thanks for the reply")
	}
}

// TestAskUserToolQuestionEvent: QuestionEvent appears before ToolResultEvent for the same ToolCallID.
func TestAskUserToolQuestionEvent(t *testing.T) {
	handler := &funcHITLHandler{
		fn: func(_ context.Context, _, _, _ string) (string, error) {
			return "answer", nil
		},
	}
	askTool := sleipnir.AskUserTool(handler)

	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("ask_user", "q-2", []byte(`{"question":"Ready?"}`)),
		sleipnirtest.TextResponse("done"),
	)
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	collector := sleipnirtest.NewEventCollector()
	if _, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName:  "agent",
		Prompt:     "go",
		Router:     defaultRouter(stub),
		ExtraTools: []sleipnir.Tool{askTool},
		Events:     collector,
	}); err != nil {
		t.Fatalf("Run: %v", err)
	}

	events := collector.Events()
	var questionIdx, toolResultIdx int = -1, -1
	for i, e := range events {
		switch ev := e.(type) {
		case sleipnir.QuestionEvent:
			if ev.QuestionID == "q-2" {
				questionIdx = i
			}
		case sleipnir.ToolResultEvent:
			if ev.ToolCallID == "q-2" {
				toolResultIdx = i
			}
		}
	}

	if questionIdx < 0 {
		t.Fatal("QuestionEvent not found")
	}
	if toolResultIdx < 0 {
		t.Fatal("ToolResultEvent not found")
	}
	if questionIdx >= toolResultIdx {
		t.Errorf("QuestionEvent (idx %d) must precede ToolResultEvent (idx %d)", questionIdx, toolResultIdx)
	}
}

// TestAskUserToolTimeout: very short HITLTimeout causes run to return ErrHITLTimeout + StopHITLTimeout.
func TestAskUserToolTimeout(t *testing.T) {
	handler := &funcHITLHandler{
		fn: func(ctx context.Context, _, _, _ string) (string, error) {
			<-ctx.Done()
			return "", ctx.Err()
		},
	}
	askTool := sleipnir.AskUserTool(handler)

	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("ask_user", "q-3", []byte(`{"question":"Slow?"}`)),
		// This second response should never be reached.
		sleipnirtest.TextResponse("done"),
	)
	h := mustNewHarness(t, sleipnir.Config{HITLTimeout: time.Millisecond})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	out, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName:  "agent",
		Prompt:     "go",
		Router:     defaultRouter(stub),
		ExtraTools: []sleipnir.Tool{askTool},
	})
	if !errors.Is(err, sleipnir.ErrHITLTimeout) {
		t.Errorf("err = %v, want ErrHITLTimeout", err)
	}
	if out.Stopped != sleipnir.StopHITLTimeout {
		t.Errorf("Stopped = %q, want StopHITLTimeout", out.Stopped)
	}
}

// TestAskUserToolCancelled: cancel outer ctx causes run to return ErrHITLCancelled + StopHITLCancelled.
func TestAskUserToolCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	ready := make(chan struct{})
	handler := &funcHITLHandler{
		fn: func(hCtx context.Context, _, _, _ string) (string, error) {
			close(ready) // signal that handler is running
			<-hCtx.Done()
			return "", hCtx.Err()
		},
	}
	askTool := sleipnir.AskUserTool(handler)

	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("ask_user", "q-4", []byte(`{"question":"Cancel me?"}`)),
		sleipnirtest.TextResponse("done"),
	)
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	errCh := make(chan error, 1)
	outCh := make(chan sleipnir.RunOutput, 1)
	go func() {
		o, e := h.Run(ctx, sleipnir.RunInput{
			AgentName:  "agent",
			Prompt:     "go",
			Router:     defaultRouter(stub),
			ExtraTools: []sleipnir.Tool{askTool},
		})
		outCh <- o
		errCh <- e
	}()

	// Wait until handler is running, then cancel.
	<-ready
	cancel()

	out := <-outCh
	err := <-errCh

	if !errors.Is(err, sleipnir.ErrHITLCancelled) {
		t.Errorf("err = %v, want ErrHITLCancelled", err)
	}
	if out.Stopped != sleipnir.StopHITLCancelled {
		t.Errorf("Stopped = %q, want StopHITLCancelled", out.Stopped)
	}
}

// TestAskUserToolNilHandler: AskUserTool(nil) returns IsError=true; run continues to StopDone.
func TestAskUserToolNilHandler(t *testing.T) {
	askTool := sleipnir.AskUserTool(nil)

	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("ask_user", "q-5", []byte(`{"question":"Anyone home?"}`)),
		sleipnirtest.TextResponse("done"),
	)
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	collector := sleipnirtest.NewEventCollector()
	out, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName:  "agent",
		Prompt:     "go",
		Router:     defaultRouter(stub),
		ExtraTools: []sleipnir.Tool{askTool},
		Events:     collector,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.Stopped != sleipnir.StopDone {
		t.Errorf("Stopped = %q, want StopDone", out.Stopped)
	}

	// The ToolResultEvent for ask_user should have IsError == true.
	toolResults := sleipnirtest.ByType[sleipnir.ToolResultEvent](collector)
	var found bool
	for _, e := range toolResults {
		if e.ToolCallID == "q-5" {
			found = true
			if !e.IsError {
				t.Errorf("ToolResultEvent.IsError = false, want true for nil handler")
			}
		}
	}
	if !found {
		t.Error("ToolResultEvent for ask_user not found")
	}
}

// TestAskUserToolParallel: two parallel ask_user calls both complete; run ends with StopDone.
func TestAskUserToolParallel(t *testing.T) {
	var (
		mu       sync.Mutex
		inflight int
		maxSeen  int
	)
	// Block both handlers until both are in-flight.
	gate := make(chan struct{})

	handler := &funcHITLHandler{
		fn: func(_ context.Context, _, _, _ string) (string, error) {
			mu.Lock()
			inflight++
			if inflight > maxSeen {
				maxSeen = inflight
			}
			mu.Unlock()

			<-gate // wait for test to release

			mu.Lock()
			inflight--
			mu.Unlock()
			return "parallel reply", nil
		},
	}
	askTool := sleipnir.AskUserTool(handler)

	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.MultiToolCallResponse(
			anyllm.ToolCall{ID: "p-1", Function: anyllm.FunctionCall{Name: "ask_user", Arguments: `{"question":"Q1"}`}},
			anyllm.ToolCall{ID: "p-2", Function: anyllm.FunctionCall{Name: "ask_user", Arguments: `{"question":"Q2"}`}},
		),
		sleipnirtest.TextResponse("all done"),
	)
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	errCh := make(chan error, 1)
	outCh := make(chan sleipnir.RunOutput, 1)
	go func() {
		o, e := h.Run(context.Background(), sleipnir.RunInput{
			AgentName:  "agent",
			Prompt:     "go",
			Router:     defaultRouter(stub),
			ExtraTools: []sleipnir.Tool{askTool},
		})
		outCh <- o
		errCh <- e
	}()

	// Wait until both handlers are in-flight, then release.
	for {
		mu.Lock()
		n := inflight
		mu.Unlock()
		if n >= 2 {
			break
		}
		runtime.Gosched()
	}
	close(gate) // release both handlers

	out := <-outCh
	err := <-errCh

	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.Stopped != sleipnir.StopDone {
		t.Errorf("Stopped = %q, want StopDone", out.Stopped)
	}
	if maxSeen < 2 {
		t.Errorf("max concurrent handlers = %d, want >= 2", maxSeen)
	}
}

// --- Chunk 16: todo tool tests ---

// TestTodoWriteAndRead: LLM calls todo_write then todo_read; read result contains the written task.
func TestTodoWriteAndRead(t *testing.T) {
	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("todo_write", "call1", []byte(`{"tasks":[{"id":"1","text":"buy milk","status":"pending"}]}`)),
		sleipnirtest.ToolCallResponse("todo_read", "call2", []byte(`{}`)),
		sleipnirtest.TextResponse("done"),
	)
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	writeTool := sleipnir.TodoWriteTool()
	readTool := sleipnir.TodoReadTool()
	collector := sleipnirtest.NewEventCollector()

	_, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName:  "agent",
		Prompt:     "go",
		Router:     defaultRouter(stub),
		ExtraTools: []sleipnir.Tool{writeTool, readTool},
		Events:     collector,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Verify the ToolResultEvent for todo_read contains the task written by todo_write.
	results := sleipnirtest.ByType[sleipnir.ToolResultEvent](collector)
	var readResultContent string
	for _, r := range results {
		if r.ToolCallID == "call2" {
			readResultContent = r.Result
		}
	}
	if !strings.Contains(readResultContent, "buy milk") {
		t.Errorf("todo_read result %q does not contain written task %q", readResultContent, "buy milk")
	}
}

// TestTodoWriteEmitsTodoEvent: todo_write emits a TodoEvent with correct contents.
func TestTodoWriteEmitsTodoEvent(t *testing.T) {
	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("todo_write", "call1", []byte(`{"tasks":[{"id":"42","text":"clean house","status":"in_progress"}]}`)),
		sleipnirtest.TextResponse("done"),
	)
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	collector := sleipnirtest.NewEventCollector()
	_, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName:  "agent",
		Prompt:     "go",
		Router:     defaultRouter(stub),
		ExtraTools: []sleipnir.Tool{sleipnir.TodoWriteTool()},
		Events:     collector,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	todoEvents := sleipnirtest.ByType[sleipnir.TodoEvent](collector)
	if len(todoEvents) != 1 {
		t.Fatalf("expected 1 TodoEvent, got %d", len(todoEvents))
	}
	ev := todoEvents[0]
	if ev.AgentName != "agent" {
		t.Errorf("TodoEvent.AgentName = %q, want %q", ev.AgentName, "agent")
	}
	if ev.ToolCallID != "call1" {
		t.Errorf("TodoEvent.ToolCallID = %q, want %q", ev.ToolCallID, "call1")
	}
	if len(ev.Todos) != 1 || ev.Todos[0].ID != "42" || ev.Todos[0].Text != "clean house" {
		t.Errorf("TodoEvent.Todos = %v, want [{42 clean house in_progress}]", ev.Todos)
	}
}

// TestTodoWriteFullReplace: second todo_write replaces the first list entirely.
func TestTodoWriteFullReplace(t *testing.T) {
	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("todo_write", "call1", []byte(`{"tasks":[{"id":"1","text":"first","status":"pending"}]}`)),
		sleipnirtest.ToolCallResponse("todo_write", "call2", []byte(`{"tasks":[{"id":"2","text":"second","status":"done"},{"id":"3","text":"third","status":"pending"}]}`)),
		sleipnirtest.ToolCallResponse("todo_read", "call3", []byte(`{}`)),
		sleipnirtest.TextResponse("done"),
	)
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 10}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	collector := sleipnirtest.NewEventCollector()
	_, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName:  "agent",
		Prompt:     "go",
		Router:     defaultRouter(stub),
		ExtraTools: []sleipnir.Tool{sleipnir.TodoWriteTool(), sleipnir.TodoReadTool()},
		Events:     collector,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// The read after second write should not contain "first".
	results := sleipnirtest.ByType[sleipnir.ToolResultEvent](collector)
	var readContent string
	for _, r := range results {
		if r.ToolCallID == "call3" {
			readContent = r.Result
		}
	}
	if strings.Contains(readContent, "first") {
		t.Errorf("todo_read after second write still contains %q (not replaced): %s", "first", readContent)
	}
	if !strings.Contains(readContent, "second") {
		t.Errorf("todo_read after second write missing %q: %s", "second", readContent)
	}
	if !strings.Contains(readContent, "third") {
		t.Errorf("todo_read after second write missing %q: %s", "third", readContent)
	}
}

// TestTodoPersistsAcrossTurns: write in turn 1, read in turn 3 → same tasks returned.
func TestTodoPersistsAcrossTurns(t *testing.T) {
	stub := sleipnirtest.NewStubProvider(t,
		// turn 1: write
		sleipnirtest.ToolCallResponse("todo_write", "w1", []byte(`{"tasks":[{"id":"99","text":"persistent","status":"pending"}]}`)),
		// turn 2: no-op tool call
		sleipnirtest.ToolCallResponse("todo_read", "r1", []byte(`{}`)),
		// turn 3: read again
		sleipnirtest.ToolCallResponse("todo_read", "r2", []byte(`{}`)),
		sleipnirtest.TextResponse("done"),
	)
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 10}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	collector := sleipnirtest.NewEventCollector()
	_, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName:  "agent",
		Prompt:     "go",
		Router:     defaultRouter(stub),
		ExtraTools: []sleipnir.Tool{sleipnir.TodoWriteTool(), sleipnir.TodoReadTool()},
		Events:     collector,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Both reads should return the written task.
	results := sleipnirtest.ByType[sleipnir.ToolResultEvent](collector)
	for _, callID := range []string{"r1", "r2"} {
		var found bool
		for _, r := range results {
			if r.ToolCallID == callID {
				found = true
				if !strings.Contains(r.Result, "persistent") {
					t.Errorf("todo_read %s result %q does not contain written task", callID, r.Result)
				}
			}
		}
		if !found {
			t.Errorf("no ToolResultEvent for todo_read callID %q", callID)
		}
	}
}

// TestTodoIsolatedByAgentName: two parallel sub-agents with different names each
// write different todos; each sub-agent's todo_read returns only its own tasks.
func TestTodoIsolatedByAgentName(t *testing.T) {
	childAStub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("todo_write", "aw1", []byte(`{"tasks":[{"id":"a1","text":"agent-a task","status":"pending"}]}`)),
		sleipnirtest.ToolCallResponse("todo_read", "ar1", []byte(`{}`)),
		sleipnirtest.TextResponse("a done"),
	)
	childBStub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("todo_write", "bw1", []byte(`{"tasks":[{"id":"b1","text":"agent-b task","status":"pending"}]}`)),
		sleipnirtest.ToolCallResponse("todo_read", "br1", []byte(`{}`)),
		sleipnirtest.TextResponse("b done"),
	)
	parentStub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.MultiToolCallResponse(
			anyllm.ToolCall{ID: "pa", Function: anyllm.FunctionCall{Name: "child-a", Arguments: `{"input":"go"}`}},
			anyllm.ToolCall{ID: "pb", Function: anyllm.FunctionCall{Name: "child-b", Arguments: `{"input":"go"}`}},
		),
		sleipnirtest.TextResponse("parent done"),
	)

	h := mustNewHarness(t, sleipnir.Config{AllowLateRegistration: true})
	extraTools := []sleipnir.Tool{sleipnir.TodoWriteTool(), sleipnir.TodoReadTool()}
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "child-a", MaxIterations: 5, Tools: extraTools}); err != nil {
		t.Fatalf("RegisterAgent child-a: %v", err)
	}
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "child-b", MaxIterations: 5, Tools: extraTools}); err != nil {
		t.Fatalf("RegisterAgent child-b: %v", err)
	}
	childATool, err := h.AgentAsTool("child-a")
	if err != nil {
		t.Fatalf("AgentAsTool child-a: %v", err)
	}
	childBTool, err := h.AgentAsTool("child-b")
	if err != nil {
		t.Fatalf("AgentAsTool child-b: %v", err)
	}
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "parent", MaxIterations: 5, Tools: []sleipnir.Tool{childATool, childBTool}}); err != nil {
		t.Fatalf("RegisterAgent parent: %v", err)
	}

	router := sleipnir.MapRouter{
		Overrides: map[string]sleipnir.ModelConfig{
			"child-a": {Provider: childAStub, Model: "stub"},
			"child-b": {Provider: childBStub, Model: "stub"},
			"parent":  {Provider: parentStub, Model: "stub"},
		},
	}
	collector := sleipnirtest.NewEventCollector()

	_, err = h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "parent",
		Prompt:    "go",
		Router:    router,
		Events:    collector,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// child-a's read should contain "agent-a task" but NOT "agent-b task"
	// child-b's read should contain "agent-b task" but NOT "agent-a task"
	results := sleipnirtest.ByType[sleipnir.ToolResultEvent](collector)
	checkRead := func(callID, agentName, wantContains, wantNotContains string) {
		t.Helper()
		for _, r := range results {
			if r.ToolCallID == callID && r.AgentName == agentName {
				if !strings.Contains(r.Result, wantContains) {
					t.Errorf("%s todo_read %q: result %q does not contain %q", agentName, callID, r.Result, wantContains)
				}
				if strings.Contains(r.Result, wantNotContains) {
					t.Errorf("%s todo_read %q: result %q unexpectedly contains %q (isolation failure)", agentName, callID, r.Result, wantNotContains)
				}
				return
			}
		}
		t.Errorf("no ToolResultEvent for callID=%q agentName=%q", callID, agentName)
	}
	checkRead("ar1", "child-a", "agent-a task", "agent-b task")
	checkRead("br1", "child-b", "agent-b task", "agent-a task")
}

// TestTodoReadEmptyList: todo_read before any todo_write returns "[]", no error.
func TestTodoReadEmptyList(t *testing.T) {
	stub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("todo_read", "r1", []byte(`{}`)),
		sleipnirtest.TextResponse("done"),
	)
	h := mustNewHarness(t, sleipnir.Config{})
	if err := h.RegisterAgent(sleipnir.AgentSpec{Name: "agent", MaxIterations: 5}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	collector := sleipnirtest.NewEventCollector()
	_, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName:  "agent",
		Prompt:     "go",
		Router:     defaultRouter(stub),
		ExtraTools: []sleipnir.Tool{sleipnir.TodoReadTool()},
		Events:     collector,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	results := sleipnirtest.ByType[sleipnir.ToolResultEvent](collector)
	var found bool
	for _, r := range results {
		if r.ToolCallID == "r1" {
			found = true
			if r.IsError {
				t.Errorf("todo_read on empty list returned IsError=true")
			}
			if r.Result != "[]" {
				t.Errorf("todo_read on empty list = %q, want %q", r.Result, "[]")
			}
		}
	}
	if !found {
		t.Error("no ToolResultEvent for todo_read r1")
	}
}

// TestRunSubAgentInputPassthrough verifies that tool-call arguments sent to a
// sub-agent are available as AgentInput.Input in its SystemPrompt function.
func TestRunSubAgentInputPassthrough(t *testing.T) {
	var capturedInput json.RawMessage

	childStub := sleipnirtest.NewStubProvider(t, sleipnirtest.TextResponse("done"))
	parentStub := sleipnirtest.NewStubProvider(t,
		sleipnirtest.ToolCallResponse("child", "tc-1", []byte(`{"task":"write report"}`)),
		sleipnirtest.TextResponse("parent done"),
	)

	h := mustNewHarness(t, sleipnir.Config{AllowLateRegistration: true})
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name:          "child",
		MaxIterations: 5,
		SystemPrompt: func(in sleipnir.AgentInput) string {
			capturedInput = in.Input
			return "child"
		},
	}); err != nil {
		t.Fatalf("RegisterAgent child: %v", err)
	}
	childTool, _ := h.AgentAsTool("child")
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name:          "parent",
		MaxIterations: 5,
		Tools:         []sleipnir.Tool{childTool},
	}); err != nil {
		t.Fatalf("RegisterAgent parent: %v", err)
	}

	router := sleipnir.MapRouter{
		Overrides: map[string]sleipnir.ModelConfig{
			"child":  {Provider: childStub, Model: "stub"},
			"parent": {Provider: parentStub, Model: "stub"},
		},
	}
	if _, err := h.Run(context.Background(), sleipnir.RunInput{
		AgentName: "parent",
		Prompt:    "go",
		Router:    router,
	}); err != nil {
		t.Fatalf("Run: %v", err)
	}

	want := `{"task":"write report"}`
	if string(capturedInput) != want {
		t.Errorf("AgentInput.Input = %q, want %q", string(capturedInput), want)
	}
}
