package sleipnir

import (
	"encoding/json"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

type AgentSpec struct {
	Name             string
	Description      string
	InputSchema      map[string]any
	SystemPrompt     func(AgentInput) string
	Tools            []Tool
	Middlewares      []Middleware
	MaxIterations    int
	MaxParallelTools int
}

type AgentInput struct {
	Prompt  string
	Input   json.RawMessage
	History []anyllm.Message
}

type AgentInfo struct {
	Name       string
	ParentName string // "" for top-level agents
	Depth      int    // 0 for top-level
	IsSubAgent bool
}
