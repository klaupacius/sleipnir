package compact

import (
	"context"
	"fmt"
	"strings"

	anyllm "github.com/mozilla-ai/any-llm-go"
	sleipnir "sleipnir.dev/sleipnir"
)

// Config configures the Compactor middleware.
type Config struct {
	// Provider to use for summarization. Typically a cheap/fast model.
	Provider anyllm.Provider
	Model    string
	// Threshold fraction of estimated context window at which to compact.
	// 0 → defaults to 0.75.
	Threshold float64
	// ContextWindow is the estimated total context size in tokens. Default: 100_000.
	ContextWindow int
}

// Compactor is a ContextRewriter that summarizes old messages when the context
// window exceeds the threshold. It skips sub-agents (they have isolated histories).
type Compactor struct {
	sleipnir.BaseMiddleware
	cfg Config
}

// NewCompactor constructs a Compactor with the given Config.
func NewCompactor(cfg Config) *Compactor { return &Compactor{cfg: cfg} }

var _ sleipnir.ContextRewriter = (*Compactor)(nil)

// RewriteBeforeLLMCall compacts the message history when the estimated token
// count exceeds the configured threshold. Sub-agent contexts are skipped.
func (c *Compactor) RewriteBeforeLLMCall(ctx context.Context, req *sleipnir.LLMRequest) error {
	if req.Agent.IsSubAgent {
		return nil
	}

	cs, ok := sleipnir.CompactStoreFrom(ctx)
	if !ok {
		return nil
	}

	contextWindow := c.cfg.ContextWindow
	if contextWindow <= 0 {
		contextWindow = 100_000
	}
	threshold := c.cfg.Threshold
	if threshold <= 0 {
		threshold = 0.75
	}

	estimated := estimateTokens(req.Messages)
	if estimated < int(float64(contextWindow)*threshold) {
		return nil
	}

	watermark := cs.GetWatermark(req.Agent.Name)
	toCompact := req.Messages[watermark:]
	if len(toCompact) < 2 {
		return nil
	}

	summary, err := c.summarize(ctx, toCompact)
	if err != nil {
		return fmt.Errorf("%w: %w", sleipnir.ErrCompactionFailed, err)
	}

	summaryMsg := anyllm.Message{
		Role:    anyllm.RoleUser,
		Content: "[Earlier context summarized]: " + summary,
	}
	req.Messages = append([]anyllm.Message{summaryMsg}, req.Messages[watermark+len(toCompact):]...)
	cs.SetWatermark(req.Agent.Name, watermark+len(toCompact))
	return nil
}

func (c *Compactor) summarize(ctx context.Context, msgs []anyllm.Message) (string, error) {
	var sb strings.Builder
	sb.WriteString("Summarize the following conversation concisely:\n\n")
	for _, m := range msgs {
		sb.WriteString(string(m.Role))
		sb.WriteString(": ")
		sb.WriteString(m.ContentString())
		sb.WriteByte('\n')
	}
	prompt := sb.String()
	params := anyllm.CompletionParams{
		Model: c.cfg.Model,
		Messages: []anyllm.Message{
			{Role: anyllm.RoleUser, Content: prompt},
		},
	}
	resp, err := c.cfg.Provider.Completion(ctx, params)
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("compact: summarizer returned no choices")
	}
	return resp.Choices[0].Message.ContentString(), nil
}

func estimateTokens(msgs []anyllm.Message) int {
	var total int
	for _, m := range msgs {
		total += len(m.ContentString()) / 4
	}
	return total
}
