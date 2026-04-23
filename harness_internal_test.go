package sleipnir

import "testing"

// Spec with MaxIterations: 0 -> stored spec has cfg.DefaultMaxIterations
func TestRegisterAgentResolvesZeros(t *testing.T) {
	h, err := NewHarness(Config{DefaultMaxIterations: 7, DefaultMaxParallelTools: 3})
	if err != nil {
		t.Fatalf("NewHarness: %v", err)
	}

	if err := h.RegisterAgent(AgentSpec{Name: "agent"}); err != nil {
		t.Fatalf("RegisterAgent: %v", err)
	}

	raw, ok := h.agents.Load("agent")
	if !ok {
		t.Fatal("agent not stored")
	}
	stored := raw.(AgentSpec)
	if stored.MaxIterations != 7 {
		t.Errorf("MaxIterations: want 7, got %d", stored.MaxIterations)
	}
	if stored.MaxParallelTools != 3 {
		t.Errorf("MaxParallelTools: want 3, got %d", stored.MaxParallelTools)
	}
}
