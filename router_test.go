package sleipnir

import (
	"context"
	"testing"
)

var _ ModelRouter = MapRouter{}

func TestMapRouterOverride(t *testing.T) {
	want := ModelConfig{Model: "claude-opus-4-5"}
	r := MapRouter{
		Default:   ModelConfig{Model: "claude-haiku-4-5"},
		Overrides: map[string]ModelConfig{"summariser": want},
	}

	got, err := r.Resolve(context.Background(), "summariser")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != want {
		t.Errorf("Resolve(%q) = %v, want %v", "summariser", got, want)
	}
}

func TestMapRouterDefault(t *testing.T) {
	want := ModelConfig{Model: "claude-haiku-4-5"}
	r := MapRouter{
		Default:   want,
		Overrides: map[string]ModelConfig{"summariser": {Model: "claude-opus-4-5"}},
	}

	got, err := r.Resolve(context.Background(), "unknown-agent")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != want {
		t.Errorf("Resolve(%q) = %v, want %v", "unknown-agent", got, want)
	}
}

func TestMapRouterEmptyOverrides(t *testing.T) {
	want := ModelConfig{Model: "claude-haiku-4-5"}
	r := MapRouter{
		Default:   want,
		Overrides: nil,
	}

	got, err := r.Resolve(context.Background(), "any-agent")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != want {
		t.Errorf("Resolve(%q) = %v, want %v", "any-agent", got, want)
	}
}
