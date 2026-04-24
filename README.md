# Sleipnir

A minimal Go multi-agent LLM orchestration harness.

> In Norse mythology, *Sleipnir* is Odin's eight-legged horse, on which he
> rides across sky and sea.

## Use of AI

This software is developed with the assistance of AI.

## Status

**Alpha.** Initial implementation complete. API may change before v1.

## What it is

- A thin canonical agent loop in readable Go.
- First-class parallel sub-agent dispatch: sub-agents are just tools.
- Typed event stream suitable for SSE, CLI, or test assertions.
- Middleware hooks for context compression, token accounting,
  observability, and retries.
- Provider-agnostic via [`any-llm-go`](https://github.com/mozilla-ai/any-llm-go)
  with runtime model/provider selection per agent.

## What it is not

A framework. No DI container, no singletons, no declarative YAML (in v1),
no persistence, no retrieval or vector-store opinions. The harness owns
no data; conversation history is an argument in and an argument out.

## Requirements

- Go 1.25+

## Install

```bash
go get sleipnir.dev/sleipnir
```

## Examples

Both examples use a scripted `demoProvider` and run without an API key.

**`examples/hello/`** — minimal working harness: one agent, one typed tool, one
custom event sink. Shows `NewHarness`, `RegisterAgent`, `NewTypedTool`, `Run`,
and the `Sink` interface.

```bash
go run ./examples/hello/
```

**`examples/orchestrator/`** — parent agent dispatching two registered
sub-agents. Demonstrates `AgentAsTool`, per-agent model routing via
`MapRouter.Overrides`, `ExtraTools` with `OmitExtraToolsInheritance`, and an
event log that collects then replays all agent lifecycle and tool events.

```bash
go run ./examples/orchestrator/
```

## Concepts

A `Harness` is the central object: you register one or more `AgentSpec` values
into it, then call `Run` to drive a single agent through the LLM loop. Each
loop iteration calls the provider, dispatches any tool calls in parallel (up to
`MaxParallelTools`), and feeds results back until the model returns a final text
response. Tools that wrap other agents give you sub-agent orchestration for
free — a sub-agent is just a `Tool`. Events stream out through a `Sink` as the
run progresses, so the caller can observe tokens, tool calls, and agent
lifecycle without polling. The LLM provider is never hardcoded: a `ModelRouter`
resolves a `ModelConfig` (provider, model, sampling parameters) for each agent
at call time, so you can route different agents to different models or providers
without changing agent code.

## API overview

### Harness

`NewHarness(cfg Config) (*Harness, error)` creates a harness from a `Config`.
`RegisterAgent(spec AgentSpec) error` adds an agent; once `Run` is called the
harness is frozen against further registrations (unless `AllowLateRegistration`
is set). `Run(ctx, RunInput) (RunOutput, error)` executes the named agent and
returns the final text, full message history, aggregate token usage, and the
stop reason. `Config` can be built directly or loaded from environment variables
with `LoadConfigFromEnv()`.

### Agents

An `AgentSpec` declares an agent's name, an optional description and input
schema (used when the agent is exposed as a tool), a `SystemPrompt` function
that receives the `AgentInput` and returns a string, a list of `Tool` values,
a list of `Middleware` values, and per-agent overrides for `MaxIterations` and
`MaxParallelTools`. Zero values for the iteration and parallelism limits fall
back to the harness-wide defaults in `Config`.

### Tools

The `Tool` interface has two methods: `Definition() ToolDefinition` (name,
description, JSON schema) and `Invoke(ctx, json.RawMessage) (ToolResult,
error)`. `NewFuncTool` constructs a tool from a raw-JSON handler when you want
full control over argument parsing. `NewTypedTool[T]` constructs a tool from a
typed handler — the JSON schema is derived automatically from the type parameter
via `invopop/jsonschema`. `ToolResult` carries a `Content` string and an
`IsError` bool; a non-nil `error` return from `Invoke` signals an
infrastructure failure that is isolated from the run, while `ToolResult{IsError:
true}` signals a structured failure that the LLM can read and react to.

### Router

`ModelRouter` is an interface with a single method: `Resolve(ctx, agentName)
(ModelConfig, error)`. `MapRouter` is a ready-made implementation: set
`Default` for the baseline model config and add per-agent entries in
`Overrides`. `ModelConfig` holds the `anyllm.Provider`, model string,
reasoning effort, temperature, and max output tokens.

### Events

All events implement the `Event` interface. The full set is:
`AgentStartEvent`, `AgentEndEvent`, `TokenEvent`, `ThinkingEvent`,
`ToolCallEvent`, `ToolResultEvent`, `QuestionEvent`, `TodoEvent`, and
`ErrorEvent`. Every event that carries an agent name also implements
`AgentNamer` (`EventAgent() string`), which lets sinks filter by agent. The
`Sink` interface has a single method, `Send(Event)`. `BufferedSink` is a
channel-backed implementation: create one with `NewBufferedSink(ctx, size)`,
pass it in `RunInput.Events`, and read events from `Events()`. Dropped events
(when the channel is full) are counted by `DroppedCount()`.

### Errors

Harness lifecycle: `ErrHarnessFrozen`, `ErrAgentNotRegistered`,
`ErrToolNameCollision`.

Run budget: `ErrIterationBudget` (loop exhausted `MaxIterations`),
`ErrTokenBudget` (cumulative usage exceeded `RunInput.MaxTotalTokens`). Both
are returned alongside a partial `RunOutput`.

Human-in-the-loop: `ErrHITLTimeout`, `ErrHITLCancelled`.

Middleware: `ErrCompactionFailed` (compactor failed; run proceeds uncompacted).

## Sub-packages

| Package | Purpose |
|---|---|
| `sleipnir.dev/sleipnir/middleware/retry` | `DefaultRetryPolicy` — exponential backoff on transient LLM errors |
| `sleipnir.dev/sleipnir/middleware/slogobs` | `SlogObserver` — structured log observer for all LLM and tool events |
| `sleipnir.dev/sleipnir/middleware/accounting` | `TokenAccountant` — per-agent and aggregate token tracking |
| `sleipnir.dev/sleipnir/middleware/compact` | `Compactor` — context-window compaction when history approaches the model limit |
| `sleipnir.dev/sleipnir/mcpadapter` | Load tools from an MCP server into any agent |
| `sleipnir.dev/sleipnir/sleipnirtest` | `StubProvider`, `EventCollector`, `StubTool` — test helpers for agent behaviour |

## License

[Apache-2.0](LICENSE).
