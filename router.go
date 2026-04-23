package sleipnir

import (
	"context"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

type ModelConfig struct {
	Provider        anyllm.Provider
	Model           string
	ReasoningEffort anyllm.ReasoningEffort
	Temperature     *float64
	MaxOutputTokens *int
}

type ModelRouter interface {
	Resolve(ctx context.Context, agentName string) (ModelConfig, error)
}

type MapRouter struct {
	Default   ModelConfig
	Overrides map[string]ModelConfig
}

func (r MapRouter) Resolve(_ context.Context, agentName string) (ModelConfig, error) {
	if cfg, ok := r.Overrides[agentName]; ok {
		return cfg, nil
	}
	return r.Default, nil
}
