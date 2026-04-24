package sleipnir_test

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
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
