package sleipnir

import (
	"context"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

type RunInput struct {
	AgentName                 string
	Prompt                    string
	Input                     any
	History                   []anyllm.Message
	Router                    ModelRouter
	Events                    Sink
	HITL                      HITLHandler
	ExtraTools                []Tool
	OmitExtraToolsInheritance bool // false (default) = sub-agents inherit ExtraTools
	MaxTotalTokens            int
}

type RunOutput struct {
	Text     string
	Messages []anyllm.Message
	Usage    Usage
	Stopped  StopReason
}

type HITLHandler interface {
	AskUser(ctx context.Context, agent, question, contextBlurb string) (string, error)
}

type Usage struct {
	InputTokens  int64
	OutputTokens int64
	TotalTokens  int64
}

type StopReason string

const (
	StopDone            StopReason = "done"
	StopIterationBudget StopReason = "iteration_budget"
	StopTokenBudget     StopReason = "token_budget"
	StopHITLTimeout     StopReason = "hitl_timeout"
	StopHITLCancelled   StopReason = "hitl_cancelled"
)
