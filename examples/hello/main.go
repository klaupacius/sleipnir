package main

import (
	"context"
	"fmt"
	"log"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"sleipnir.dev/sleipnir"
)

// demoProvider returns scripted responses so the examples run without
// an API key.
type demoProvider struct {
	turns []*anyllm.ChatCompletion
	n int
}

func (p *demoProvider) Name() string {return "demo"}

func (p *demoProvider) Completion(_ context.Context, _ anyllm.CompletionParams) (*anyllm.ChatCompletion, error) {
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

// printSink implements sleipnir.Sink and prints tool events to stdout.
type printSink struct{}

func (printSink) Send(e sleipnir.Event) {
	switch ev := e.(type) {
	case sleipnir.ToolCallEvent:
		fmt.Printf("  -> tool call:   %s(%s)\n", ev.ToolName, string(ev.Args))
	case sleipnir.ToolResultEvent:
		fmt.Printf("  <- tool result: %s\n", ev.Result)
	}
}

type greetArgs struct {
	Name string `json:"name"`
}

func main() {
	ctx := context.Background()

	// Scripted provider: turn 1 calls the tool, turn 2 gives the final answer.
	provider := &demoProvider{
		turns: []*anyllm.ChatCompletion{
			{
				Choices: []anyllm.Choice{{
					Message: anyllm.Message{
						Role: anyllm.RoleAssistant,
						ToolCalls: []anyllm.ToolCall{{
							ID: "call_1",
							Function: anyllm.FunctionCall{
								Name: "get_greeting",
								Arguments: `{"name":"Alice"}`,
							},
						}},
					},
					FinishReason: anyllm.FinishReasonToolCalls,
				}},
				Usage: &anyllm.Usage{PromptTokens: 20, CompletionTokens: 10},
			},
			{
				Choices: []anyllm.Choice{{
					Message: anyllm.Message{
						Role: anyllm.RoleAssistant,
						Content: `The greeting for Alice is: "Hello, Alice!"`,
					},
					FinishReason: anyllm.FinishReasonStop,
				}},
				Usage: &anyllm.Usage{PromptTokens: 30, CompletionTokens: 8},
			},
		},
	}

	h, err := sleipnir.NewHarness(sleipnir.Config{})
	if err != nil {
		log.Fatal(err)
	}

	greetTool, err := sleipnir.NewTypedTool[greetArgs](
		"get_greeting",
		"Returns a greeting for a given name.",
		func(_ context.Context, args greetArgs) (sleipnir.ToolResult, error) {
			return sleipnir.ToolResult{Content: "Hello, " + args.Name + "!"}, nil
		},
	)
	if err := h.RegisterAgent(sleipnir.AgentSpec{
		Name: "greeter",
		SystemPrompt: func(_ sleipnir.AgentInput) string {
			return "You are a helpful assistant. Use get_greeting to greet people by name."
		},
		Tools: []sleipnir.Tool{greetTool},
	}); err != nil {
		log.Fatal(err)
	}

	fmt.Println("Running greeter agent...")
	out, err := h.Run(ctx, sleipnir.RunInput{
		AgentName: "greeter",
		Prompt: "What is the greeting for Alice?",
		Router: sleipnir.MapRouter{
			Default: sleipnir.ModelConfig{
				Provider: provider,
				Model: "demo-model",
			},
		},
		Events: printSink{},
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Agent: %s\n", out.Text)
	fmt.Printf("Tokens: input=%d output=%d\n", out.Usage.InputTokens, out.Usage.OutputTokens)
}