package sleipnir_test

import (
	"context"
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
