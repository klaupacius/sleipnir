// orchestrator demonstrates a parent agent dispatching two sub-agents via
// AgentAsTool, with per-agent model routing, ExtraTools isolation, and
// event collection for replay. Run with `go run ./examples/orchestrator/`.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"sleipnir.dev/sleipnir"
)

// demoProvider returns scripted responses so the example runs without an API
// key. Each agent gets its own instance (routed via MapRouter.Overrides) so
// their response sequences stay independent. The mutex is required because
// dispatchInParallel may invoke sub-agents concurrently from a single turn.
type demoProvider struct {
	mu    sync.Mutex
	turns []*anyllm.ChatCompletion
	n     int
}

func (p *demoProvider) Name() string { return "demo" }

func (p *demoProvider) Completion(_ context.Context, _ anyllm.CompletionParams) (*anyllm.ChatCompletion, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.n >= len(p.turns) {
		return nil, fmt.Errorf("demoProvider: ran out of scripted responses")
	}
	r := p.turns[p.n]
	p.n++
	return r, nil
}

func (p *demoProvider) CompletionStream(_ context.Context, _ anyllm.CompletionParams) (<-chan anyllm.ChatCompletionChunk, <-chan error) {
	panic("demoProvider: streaming not implemented")
}

// eventLog implements sleipnir.Sink and accumulates events for replay after
// the run completes. This is the same collect-then-replay pattern provided by
// sleipnirtest.EventCollector in test code; it is reimplemented here to avoid
// importing the testing package into this binary.
type eventLog struct {
	mu      sync.Mutex
	records []sleipnir.Event
}

func (l *eventLog) Send(e sleipnir.Event) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.records = append(l.records, e)
}

// printAll replays all collected events to stdout in arrival order.
// Prefixing each line with the emitting agent name makes interleaved
// output from parent and sub-agents easy to follow.
func (l *eventLog) printAll() {
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, e := range l.records {
		switch ev := e.(type) {
		case sleipnir.AgentStartEvent:
			if ev.ParentName != "" {
				fmt.Printf("  [%s] start (parent: %s)\n", ev.AgentName, ev.ParentName)
			} else {
				fmt.Printf("  [%s] start\n", ev.AgentName)
			}
		case sleipnir.AgentEndEvent:
			fmt.Printf("  [%s] end   stopped=%s tokens=%d\n", ev.AgentName, ev.Stopped, ev.Usage.TotalTokens)
		case sleipnir.ToolCallEvent:
			fmt.Printf("  [%s] ->    %s(%s)\n", ev.AgentName, ev.ToolName, string(ev.Args))
		case sleipnir.ToolResultEvent:
			if ev.IsError {
				fmt.Printf("  [%s] <-    ERROR: %s\n", ev.AgentName, ev.Result)
			} else {
				fmt.Printf("  [%s] <-    %s\n", ev.AgentName, ev.Result)
			}
		}
	}
}

// searchArgs is the input schema for the researcher's search tool.
type searchArgs struct {
	Query string `json:"query"`
}

// subAgentArgs is the call signature the coordinator uses when dispatching
// researcher or writer. AgentSpec.InputSchema is set to match this shape.
type subAgentArgs struct {
	Task string `json:"task"`
}

// subAgentSchema returns the JSON schema for subAgentArgs.
func subAgentSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"task": map[string]any{
				"type":        "string",
				"description": "The work to perform.",
			},
		},
		"required": []string{"task"},
	}
}

func main() {
	ctx := context.Background()

	// coordinatorProvider scripts 4 turns:
	//   1. call timestamp (an ExtraTool — only coordinator has it)
	//   2. call researcher sub-agent
	//   3. call writer sub-agent
	//   4. return the final report
	coordinatorProvider := &demoProvider{
		turns: []*anyllm.ChatCompletion{
			{
				Choices: []anyllm.Choice{{
					Message: anyllm.Message{
						Role: anyllm.RoleAssistant,
						ToolCalls: []anyllm.ToolCall{{
							ID: "call_ts",
							Function: anyllm.FunctionCall{
								Name:      "timestamp",
								Arguments: `{}`,
							},
						}},
					},
					FinishReason: anyllm.FinishReasonToolCalls,
				}},
				Usage: &anyllm.Usage{PromptTokens: 30, CompletionTokens: 10, TotalTokens: 40},
			},
			{
				Choices: []anyllm.Choice{{
					Message: anyllm.Message{
						Role: anyllm.RoleAssistant,
						ToolCalls: []anyllm.ToolCall{{
							ID: "call_research",
							Function: anyllm.FunctionCall{
								Name:      "researcher",
								Arguments: `{"task":"research Go generics"}`,
							},
						}},
					},
					FinishReason: anyllm.FinishReasonToolCalls,
				}},
				Usage: &anyllm.Usage{PromptTokens: 50, CompletionTokens: 15, TotalTokens: 65},
			},
			{
				Choices: []anyllm.Choice{{
					Message: anyllm.Message{
						Role: anyllm.RoleAssistant,
						ToolCalls: []anyllm.ToolCall{{
							ID: "call_write",
							Function: anyllm.FunctionCall{
								Name:      "writer",
								Arguments: `{"task":"write report on Go generics, dated 2026-04-24"}`,
							},
						}},
					},
					FinishReason: anyllm.FinishReasonToolCalls,
				}},
				Usage: &anyllm.Usage{PromptTokens: 80, CompletionTokens: 15, TotalTokens: 95},
			},
			{
				Choices: []anyllm.Choice{{
					Message: anyllm.Message{
						Role:    anyllm.RoleAssistant,
						Content: "Report (2026-04-24): Go generics, introduced in 1.18, enable type-safe reusable algorithms.",
					},
					FinishReason: anyllm.FinishReasonStop,
				}},
				Usage: &anyllm.Usage{PromptTokens: 120, CompletionTokens: 20, TotalTokens: 140},
			},
		},
	}

	// researcherProvider scripts 2 turns: call search, then return findings.
	researcherProvider := &demoProvider{
		turns: []*anyllm.ChatCompletion{
			{
				Choices: []anyllm.Choice{{
					Message: anyllm.Message{
						Role: anyllm.RoleAssistant,
						ToolCalls: []anyllm.ToolCall{{
							ID: "call_search",
							Function: anyllm.FunctionCall{
								Name:      "search",
								Arguments: `{"query":"Go generics"}`,
							},
						}},
					},
					FinishReason: anyllm.FinishReasonToolCalls,
				}},
				Usage: &anyllm.Usage{PromptTokens: 20, CompletionTokens: 8, TotalTokens: 28},
			},
			{
				Choices: []anyllm.Choice{{
					Message: anyllm.Message{
						Role:    anyllm.RoleAssistant,
						Content: "Go 1.18 introduced generics, allowing type parameters in functions and types.",
					},
					FinishReason: anyllm.FinishReasonStop,
				}},
				Usage: &anyllm.Usage{PromptTokens: 30, CompletionTokens: 12, TotalTokens: 42},
			},
		},
	}

	// writerProvider scripts 1 turn: return the formatted report directly.
	writerProvider := &demoProvider{
		turns: []*anyllm.ChatCompletion{
			{
				Choices: []anyllm.Choice{{
					Message: anyllm.Message{
						Role:    anyllm.RoleAssistant,
						Content: "Go Generics Report (2026-04-24): Since 1.18, Go supports type parameters, enabling type-safe reusable code without reflection.",
					},
					FinishReason: anyllm.FinishReasonStop,
				}},
				Usage: &anyllm.Usage{PromptTokens: 40, CompletionTokens: 18, TotalTokens: 58},
			},
		},
	}

	// NewHarness with default Config. All agents are registered before Run is
	// called, so AllowLateRegistration is not needed.
	h, err := sleipnir.NewHarness(sleipnir.Config{})
	if err != nil {
		log.Fatal(err)
	}

	// Register sub-agents first so AgentAsTool can reference them.

	searchTool, err := sleipnir.NewTypedTool[searchArgs](
		"search",
		"Searches for information on a topic.",
		func(_ context.Context, args searchArgs) (sleipnir.ToolResult, error) {
			return sleipnir.ToolResult{Content: "search results for: " + args.Query}, nil
		},
	)
	if err != nil {
		log.Fatal(err)
	}

	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name:        "researcher",
		Description: "Researches a topic and returns key findings.",
		InputSchema: subAgentSchema(),
		SystemPrompt: func(in sleipnir.AgentInput) string {
			// in.Input holds the raw tool-call arguments from the coordinator.
			// Parse them to tailor the system prompt to the specific task.
			var req subAgentArgs
			json.Unmarshal(in.Input, &req) //nolint:errcheck — demo; invalid JSON yields zero value
			if req.Task != "" {
				return "You are a research assistant. Your task: " + req.Task
			}
			return "You are a research assistant. Use the search tool to find information."
		},
		Tools: []sleipnir.Tool{searchTool},
	}); err != nil {
		log.Fatal(err)
	}

	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name:        "writer",
		Description: "Writes a formatted report from provided information.",
		InputSchema: subAgentSchema(),
		SystemPrompt: func(in sleipnir.AgentInput) string {
			var req subAgentArgs
			json.Unmarshal(in.Input, &req) //nolint:errcheck — demo; invalid JSON yields zero value
			if req.Task != "" {
				return "You are a technical writer. Your task: " + req.Task
			}
			return "You are a technical writer. Write clear, concise reports."
		},
	}); err != nil {
		log.Fatal(err)
	}

	// AgentAsTool wraps a registered agent as a Tool. The tool's name,
	// description, and input schema come from the sub-agent's AgentSpec.
	researcherTool, err := h.AgentAsTool("researcher")
	if err != nil {
		log.Fatal(err)
	}
	writerTool, err := h.AgentAsTool("writer")
	if err != nil {
		log.Fatal(err)
	}

	// timestamp is passed as an ExtraTool so it is available to the
	// coordinator at runtime without being baked into its AgentSpec.
	timestampTool := sleipnir.NewFuncTool(
		"timestamp",
		"Returns today's date as YYYY-MM-DD.",
		nil,
		func(_ context.Context, _ json.RawMessage) (sleipnir.ToolResult, error) {
			return sleipnir.ToolResult{Content: "2026-04-24"}, nil
		},
	)

	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name: "coordinator",
		SystemPrompt: func(_ sleipnir.AgentInput) string {
			return "You are a research coordinator. Use researcher and writer to produce a report. Use timestamp for today's date."
		},
		Tools: []sleipnir.Tool{researcherTool, writerTool},
	}); err != nil {
		log.Fatal(err)
	}

	var events eventLog

	fmt.Println("Running coordinator agent...")
	fmt.Println()

	out, err := h.Run(ctx, sleipnir.RunInput{
		AgentName: "coordinator",
		Prompt:    "Research Go generics and write a short dated report.",
		Router: sleipnir.MapRouter{
			// Route each agent to its own scripted provider so their response
			// sequences stay independent regardless of dispatch order.
			Overrides: map[string]sleipnir.ModelConfig{
				"coordinator": {Provider: coordinatorProvider, Model: "demo-model"},
				"researcher":  {Provider: researcherProvider, Model: "demo-model"},
				"writer":      {Provider: writerProvider, Model: "demo-model"},
			},
		},
		// ExtraTools are available to the coordinator at this run only.
		// OmitExtraToolsInheritance: true ensures that when the coordinator
		// dispatches researcher or writer, neither receives timestampTool.
		ExtraTools:                []sleipnir.Tool{timestampTool},
		OmitExtraToolsInheritance: true,
		Events:                    &events,
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Event log:")
	events.printAll()
	fmt.Println()
	fmt.Printf("Report:  %s\n", out.Text)
	fmt.Printf("Stopped: %s\n", out.Stopped)
	fmt.Printf("Tokens:  input=%d output=%d\n", out.Usage.InputTokens, out.Usage.OutputTokens)
}
