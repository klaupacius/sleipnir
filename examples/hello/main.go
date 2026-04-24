// hello demonstrates the minimal Sleipnir setup: one agent, one typed tool,
// and a custom event sink. Run it with `go run ./examples/hello/` — no API
// key required because demoProvider supplies scripted responses.
package main

import (
	"context"
	"fmt"
	"log"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"sleipnir.dev/sleipnir"
)

// demoProvider is a stand-in for a real LLM provider. In production you would
// use one from the any-llm-go library, for example:
//
//	provider, err := openai.New(anyllm.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
//
// Here we script two turns so the example is self-contained:
//
//	turn 1 — model calls the get_greeting tool
//	turn 2 — model incorporates the result and returns a final answer
type demoProvider struct {
	turns []*anyllm.ChatCompletion
	n     int
}

func (p *demoProvider) Name() string { return "demo" }

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

// printSink implements sleipnir.Sink by printing tool activity to stdout.
// A Sink receives every Event emitted during a run; switch on the concrete
// type to handle only the events you care about.
type printSink struct{}

func (printSink) Send(e sleipnir.Event) {
	switch ev := e.(type) {
	case sleipnir.ToolCallEvent:
		fmt.Printf("  -> tool call:   %s(%s)\n", ev.ToolName, string(ev.Args))
	case sleipnir.ToolResultEvent:
		fmt.Printf("  <- tool result: %s\n", ev.Result)
	}
}

// greetArgs is the input type for the get_greeting tool.
// NewTypedTool[T] reflects the JSON schema from this struct automatically —
// no need to write it by hand.
type greetArgs struct {
	Name string `json:"name"`
}

func main() {
	ctx := context.Background()

	provider := &demoProvider{
		turns: []*anyllm.ChatCompletion{
			// Turn 1: the model calls get_greeting.
			{
				Choices: []anyllm.Choice{{
					Message: anyllm.Message{
						Role: anyllm.RoleAssistant,
						ToolCalls: []anyllm.ToolCall{{
							ID: "call_1",
							Function: anyllm.FunctionCall{
								Name:      "get_greeting",
								Arguments: `{"name":"Alice"}`,
							},
						}},
					},
					FinishReason: anyllm.FinishReasonToolCalls,
				}},
				Usage: &anyllm.Usage{PromptTokens: 20, CompletionTokens: 10},
			},
			// Turn 2: the model incorporates the tool result and answers.
			{
				Choices: []anyllm.Choice{{
					Message: anyllm.Message{
						Role:    anyllm.RoleAssistant,
						Content: `The greeting for Alice is: "Hello, Alice!"`,
					},
					FinishReason: anyllm.FinishReasonStop,
				}},
				Usage: &anyllm.Usage{PromptTokens: 30, CompletionTokens: 8},
			},
		},
	}

	// NewHarness validates Config and applies defaults. A zero-value Config is valid.
	h, err := sleipnir.NewHarness(sleipnir.Config{})
	if err != nil {
		log.Fatal(err)
	}

	// NewTypedTool[T] builds a Tool whose JSON schema is derived from T.
	greetTool, err := sleipnir.NewTypedTool[greetArgs](
		"get_greeting",
		"Returns a greeting for the given name.",
		func(_ context.Context, args greetArgs) (sleipnir.ToolResult, error) {
			return sleipnir.ToolResult{Content: "Hello, " + args.Name + "!"}, nil
		},
	)
	if err != nil {
		log.Fatal(err)
	}

	// RegisterAgent binds the agent's name, system prompt, and tools to the harness.
	// SystemPrompt is a function so it can inspect the incoming AgentInput when needed.
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

	// Run executes the agent loop: send prompt → receive response → dispatch tools → repeat.
	// MapRouter.Default routes every agent to the same provider and model.
	out, err := h.Run(ctx, sleipnir.RunInput{
		AgentName: "greeter",
		Prompt:    "What is the greeting for Alice?",
		Router: sleipnir.MapRouter{
			Default: sleipnir.ModelConfig{
				Provider: provider,
				Model:    "demo-model",
			},
		},
		Events: printSink{},
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Agent:   %s\n", out.Text)
	// out.Stopped explains why the loop ended. StopDone means the model returned
	// a final answer with no pending tool calls. Other values (StopIterationBudget,
	// StopTokenBudget) indicate the run was cut short.
	fmt.Printf("Stopped: %s\n", out.Stopped)
	fmt.Printf("Tokens:  input=%d output=%d\n", out.Usage.InputTokens, out.Usage.OutputTokens)
}
