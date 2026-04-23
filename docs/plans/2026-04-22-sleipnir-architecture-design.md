# Sleipnir Architecture Design

**Date:** 2026-04-22
**Status:** Approved (design phase)
**Module path:** `sleipnir.dev/sleipnir`
**Repository:** `github.com/klaupacius/sleipnir`
**License:** Apache-2.0

---

## 1. Identity

Sleipnir is a small Go library providing a multi-agent LLM orchestration harness.
It is a Pydantic AI replacement for the Urd SaaS agent tier, with native support
for Claude-Code-style parallel sub-agent dispatch. Released open-source from day
one.

**What it is:** a thin canonical agent loop; a typed event stream; first-class
parallel sub-agents; middleware hooks; a provider-agnostic LLM boundary via
[`any-llm-go`](https://github.com/mozilla-ai/any-llm-go).

**What it is not:** a framework. No DI container, no registry singletons, no
declarative YAML (v1), no persistence, no opinionated storage or retrieval.
The harness owns no data.

**First consumer:** Urd SaaS, an agentic research assistant for Canadian labour
arbitration case law. Urd's current Python implementation (Pydantic AI + FastAPI
+ SSE + MCP tools + SQLite sessions + mid-turn clarifying HITL) defines the
minimum feature surface Sleipnir must support.

---

## 2. Design principles

1. **Twelve-Factor-practicable.** A programmatic `Config` struct *and* a
   `LoadConfigFromEnv()` helper. `SLEIPNIR_*` env-var namespace. Structured
   logging via `log/slog` to stdout. Stateless share-nothing harness.
   `context.Context` propagated through every call for cancellation and
   deadlines.
2. **Thin orchestration.** One canonical agent loop lives in the harness.
   Authors plug in by supplying an `AgentSpec`, not by reimplementing the
   loop. The loop is readable Go — no reflection-driven magic.
3. **Sub-agents are tools.** From a parent agent's perspective, a sub-agent
   is indistinguishable from any other tool. Parallel dispatch happens
   naturally when the LLM emits multiple `tool_use` blocks in one turn.
4. **Strict sub-agent isolation.** Each sub-agent has its own message
   history, its own tool list, and its own resolved model. The parent sees
   only the sub-agent's final text as a `tool_result`. No context is shared
   implicitly.
5. **Middleware for cross-cutting concerns.** Context compression, token
   accounting, observability, and retry are all middleware. Shipped defaults;
   swappable per deployment.
6. **Harness owns no data.** Conversation history is an argument in /
   argument out. Caller owns persistence. Tool and MCP-client lifecycles are
   caller-owned.
7. **Provider-agnostic.** LLM access exclusively through `any-llm-go`.
   Model/provider chosen at `Run()` time via a `ModelRouter`, with per-agent
   overrides (Claude-Code-style).

---

## 3. Architecture overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Caller (e.g., urd-agent)                 │
│   owns: session DB · MCP client · Provider · router config  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Run(ctx, RunInput)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                          Harness                            │
│                                                             │
│  canonical agent loop:                                      │
│     resolve model (Router) → LLM call (any-llm-go) →        │
│     parse tool_use blocks → dispatch tools in parallel →    │
│       · leaf tool:      invoke via Tool.Invoke              │
│       · sub-agent tool: recurse into same loop, isolated    │
│     append tool_result → repeat until final text            │
│                                                             │
│  middleware chain wraps each LLM call + tool call:          │
│     ContextRewriter · LLMObserver · ToolObserver ·          │
│     RetryPolicy · TokenAccountant                           │
│                                                             │
│  events stream out via EventSink (typed Go channel)         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                  RunOutput{Messages, Usage, ...}
                  + event stream consumed by caller
                  (urd-agent maps Event → SSE dict)
```

---

## 4. Core public API

```go
package sleipnir

// ---- Harness ---------------------------------------------------------------
type Harness struct { /* unexported */ }

func New(cfg Config) (*Harness, error)
func (h *Harness) RegisterAgent(spec AgentSpec) error
func (h *Harness) Run(ctx context.Context, in RunInput) (RunOutput, error)
func (h *Harness) AgentAsTool(name string) (Tool, error)

// ---- AgentSpec -------------------------------------------------------------
type AgentSpec struct {
    Name             string
    Description      string                           // shown to a parent agent
    InputSchema      map[string]any                   // JSON Schema for inputs
    SystemPrompt     func(in AgentInput) string
    Tools            []Tool                           // includes sub-agents-as-tools
    MaxIterations    int                              // 0 → Config default
    MaxParallelTools int                              // 0 → Config default
    EnableAskUser    bool                             // opt-in HITL
    EnableTodoList   bool                             // opt-in planning tool
}

// ---- RunInput / RunOutput --------------------------------------------------
type RunInput struct {
    AgentName      string
    Prompt         string
    Input          any                                // marshalled against InputSchema
    History        []anyllm.Message                   // caller-owned
    Router         ModelRouter                        // required
    Events         EventSink                          // nil = discard
    Deadline       time.Time                          // zero = ctx-only
    MaxTotalTokens int                                // run-wide; 0 = unlimited
}

type RunOutput struct {
    Text     string                                   // final assistant text
    Messages []anyllm.Message                         // updated history
    Usage    Usage                                    // aggregated parent + subs
    Stopped  StopReason                               // Done | IterationBudget | ...
}

// ---- ModelRouter -----------------------------------------------------------
type ModelConfig struct {
    Provider        anyllm.Provider
    Model           string
    ReasoningEffort anyllm.ReasoningEffort
    Temperature     *float64
    MaxOutputTokens *int
}
type ModelRouter interface { Resolve(agentName string) ModelConfig }

type MapRouter struct {
    Default   ModelConfig
    Overrides map[string]ModelConfig                  // keyed by AgentSpec.Name
}

// ---- Tool ------------------------------------------------------------------
type Tool interface {
    Definition() ToolDefinition
    Invoke(ctx context.Context, args json.RawMessage) (ToolResult, error)
}
type ToolDefinition struct {
    Name        string
    Description string
    InputSchema map[string]any
}
type ToolResult struct {
    Content string
    IsError bool
}

// Helpers:
func NewTypedTool[T any](name, desc string,
    fn func(context.Context, T) (ToolResult, error)) Tool
func NewFuncTool(name, desc string, schema map[string]any,
    fn func(context.Context, json.RawMessage) (ToolResult, error)) Tool
```

---

## 5. Canonical agent loop

Implemented once in `harness.go`. Every agent — parent or sub-agent —
executes the same loop.

```text
for i := 0; i < spec.MaxIterations; i++ {
    params := assembleParams(spec, history, router.Resolve(spec.Name))
    applyBefore(middlewares, ctx, &params)
    resp, err := providerCompletionWithRetry(ctx, params)
    applyAfter(middlewares, ctx, &resp, err)
    if err != nil { return err }

    history = append(history, resp.Choices[0].Message)
    emitTokenAndToolCallEvents(resp, sink)

    toolCalls := resp.Choices[0].Message.ToolCalls
    if len(toolCalls) == 0 {
        return RunOutput{Text: resp.Choices[0].Message.Content,
                         Messages: history,
                         Usage: accum.Usage(),
                         Stopped: StopDone}
    }

    results := dispatchInParallel(ctx, toolCalls, spec, h, sink,
                                  spec.MaxParallelTools)
    history = append(history, toolResultMessages(results)...)
}
return ErrIterationBudget
```

Dispatch rules:

- Each `ToolCall` is resolved against `spec.Tools` by name. If the matched
  `Tool` is a sub-agent wrapper (from `AgentAsTool`), the harness recurses
  into `Run` with a fresh `RunInput` (new goroutine, isolated history,
  parent-tagged events). Otherwise, `Tool.Invoke` runs in a goroutine.
- Parallelism bounded by `MaxParallelTools` via a semaphore.
- Errors from one tool do **not** cancel siblings; they return as
  `IsError: true` tool_results so the LLM can react.
- `context.Context` cancellation cascades: cancelling a Run ctx cancels all
  in-flight LLM calls, tools, and sub-agents cooperatively.

---

## 6. Events and streaming

```go
type EventSink interface { Send(Event) }

type Event interface{ eventMarker() }

type AgentStartEvent  struct { AgentName, ParentName string }
type AgentEndEvent    struct { AgentName string; Usage Usage; Stopped StopReason }
type TokenEvent       struct { AgentName, Text string }
type ThinkingEvent    struct { AgentName, Text string }
type ToolCallEvent    struct { AgentName, ToolCallID, ToolName string; Args json.RawMessage }
type ToolResultEvent  struct { AgentName, ToolCallID string; Result string; IsError bool }
type QuestionEvent    struct { AgentName, Question string; ReplyCh chan<- string }
type TodoEvent        struct { AgentName string; Todos []TodoItem }
type ErrorEvent       struct { AgentName string; Err error }
```

- Sub-agent events flow through the **same** sink, tagged by `AgentName`
  (and `ParentName` on start).
- Consumer filters/rolls up; harness stays policy-free.
- Urd's API layer maps each `Event` variant to a `text/event-stream` payload
  matching the existing SSE contract (`token`, `thinking`, `tool_call`,
  `tool_done`, `question`, `done`, `error`).

---

## 7. Middleware

A middleware value may implement any subset of these interfaces; the
harness discovers capabilities via type assertion and calls what's
implemented. Order is preserved from registration.

```go
type ContextRewriter interface {
    RewriteBeforeLLMCall(ctx context.Context, req *LLMRequest) error
}
type LLMObserver interface {
    OnLLMCall(ctx context.Context, req *LLMRequest, resp *LLMResponse, err error)
}
type ToolObserver interface {
    OnToolCall(ctx context.Context, call *ToolCall, result *ToolResult, err error)
}
type RetryPolicy interface {
    ShouldRetry(ctx context.Context, attempt int, err error) (retry bool, backoff time.Duration)
}
```

### Shipped middlewares (subpackages)

| Subpackage | Provides | Role |
|---|---|---|
| `middleware/compact` | `Compactor` (`ContextRewriter`) | LLM-based summarization of oldest messages when prompt exceeds `SLEIPNIR_COMPACT_THRESHOLD` (default 75% of model context). Summarizer model from `SLEIPNIR_COMPACT_MODEL` (default `claude-haiku-4-5`). Off for sub-agents. |
| `middleware/accounting` | `TokenAccountant` (`LLMObserver`) | Aggregates `anyllm.Usage` across parent + subs; surfaced via `RunOutput.Usage`. |
| `middleware/slogobs` | `SlogObserver` (`LLMObserver` + `ToolObserver`) | Structured logs. Level + format from env. |
| `middleware/retry` | `DefaultRetryPolicy` (`RetryPolicy`) | LLM-call retries: exponential backoff + jitter, max 3 attempts (`SLEIPNIR_MAX_LLM_RETRIES`). Retries `anyllm.ErrRateLimit` and `anyllm.ErrProvider`. Does *not* retry `ErrInvalidRequest`, `ErrContextLength`, `ErrAuthentication`, `ErrContentFilter`, `ErrModelNotFound`, `ErrInsufficientFunds`. Tool errors are never retried by the harness. |

OpenTelemetry observer arrives as `middleware/otelobs` in v2.

---

## 8. Tools

### Authoring

- `NewTypedTool[T](name, desc, fn)` — input struct `T` with `json:` and
  `jsonschema:` tags; schema reflected at construction time via
  [`invopop/jsonschema`](https://github.com/invopop/jsonschema). Primary
  path for Go-native tools.
- `NewFuncTool(name, desc, schema, fn)` — raw `json.RawMessage` in,
  `ToolResult` out, hand-written schema. Escape hatch for dynamic /
  union schemas.

### MCP adapter

Separate package `sleipnir.dev/sleipnir/mcpadapter`. Depends on
`github.com/modelcontextprotocol/go-sdk`. Core does not depend on MCP.

```go
tools, err := mcpadapter.LoadTools(ctx, mcpClient,
    mcpadapter.WithPrefix("urd_"),   // optional; e.g. urd_search_fts
)
spec.Tools = append(spec.Tools, tools...)
```

- Wraps each MCP tool into a Sleipnir `Tool`; `Invoke` forwards to MCP
  `tools/call`.
- Caller owns the MCP client lifecycle.

### Name collisions

`RegisterAgent` validates `AgentSpec.Tools` at registration time. Duplicate
names → hard error (`ErrToolNameCollision`). Namespacing is the caller's
responsibility via `WithPrefix`; the harness does not silently rename.

---

## 9. Human-in-the-loop (`ask_user`)

Opt-in per agent (`AgentSpec.EnableAskUser = true`). The harness synthesizes
an `ask_user` tool in the agent's tool list with fixed schema:

```json
{"question": "string", "context": "string (optional)"}
```

Execution:

1. LLM calls `ask_user`.
2. Harness emits `QuestionEvent{AgentName, Question, ReplyCh chan<- string}`.
3. Harness blocks on `ReplyCh`, `ctx.Done()`, or `SLEIPNIR_HITL_TIMEOUT`
   (default 30 min).
4. On receive → string becomes `tool_result` content; loop continues.
5. On cancel / timeout → `Run` returns `ErrHITLCancelled` /
   `ErrHITLTimeout`.

Sub-agents may opt in; `QuestionEvent.AgentName` lets the consumer route
replies correctly.

---

## 10. Todo / planning tool

Opt-in per agent (`AgentSpec.EnableTodoList = true`). Synthesizes two
built-in tools with **full-replace** semantics:

```text
todo_write: {tasks: [{id: string, text: string,
                      status: "pending"|"in_progress"|"done"}]}
    replaces the agent's current todo list.
todo_read:  {}
    returns the current todo list.
```

- State lives in an in-memory list scoped to the agent *instance* that
  created it. Parent and each sub-agent get their own list.
- On every update, `TodoEvent{AgentName, Todos}` is emitted.
- **Not persisted by the harness.** Callers who want session-scoped todos
  observe `TodoEvent` and write to their own store.

Full-replace (vs. granular `todo_update_status` / `todo_remove`) was chosen
for: lower prompt overhead, natural idempotence, fewer LLM reasoning paths,
fewer harness edge cases. Validated pattern from Claude Code's `TodoWrite`.

---

## 11. Budgets, limits, cancellation

| Knob | Where | Default |
|---|---|---|
| `MaxIterations` | `AgentSpec` (per agent) | 20 (Config) |
| `MaxParallelTools` | `AgentSpec` (per agent) | 8 (Config) |
| `Deadline` | `RunInput` (per run) | none (ctx-only) |
| `MaxTotalTokens` | `RunInput` (per run, parent + subs) | 0 (unlimited) |

Typed errors: `ErrIterationBudget`, `ErrTokenBudget`, `ErrDeadline`,
`ErrHITLTimeout`, `ErrHITLCancelled`, `ErrCompactionFailed`,
`ErrAgentNotRegistered`, `ErrToolNameCollision`.

`context.Context` cancellation cascades. Sub-agents always inherit the
parent's ctx.

---

## 12. Configuration (twelve-factor)

```go
type Config struct {
    DefaultMaxIterations    int
    DefaultMaxParallelTools int
    CompactThreshold        int            // fraction * 10000 (7500 = 75%)
    CompactModel            string
    MaxLLMRetries           int
    HITLTimeout             time.Duration
    LogLevel                slog.Level
    LogFormat               string         // "json" | "text"
    Middlewares             []any
}

func LoadConfigFromEnv() (Config, error)
```

Environment variables read by `LoadConfigFromEnv`:

```text
SLEIPNIR_MAX_ITERATIONS
SLEIPNIR_MAX_PARALLEL_TOOLS
SLEIPNIR_COMPACT_THRESHOLD
SLEIPNIR_COMPACT_MODEL
SLEIPNIR_MAX_LLM_RETRIES
SLEIPNIR_HITL_TIMEOUT
SLEIPNIR_LOG_LEVEL
SLEIPNIR_LOG_FORMAT
```

Missing vars fall back to documented defaults. No file reads. No globals.
Both `Config{}` (tests) and `LoadConfigFromEnv()` (production) are
first-class.

**Review Note:** Consider if `Middlewares` belongs in individual agent configuration rather than the main app config. Or maybe both, with agent config overriding the app config defaults?

---

## 13. Testing support (`sleipnirtest/`)

| Helper | Purpose |
|---|---|
| `StubProvider` | Implements `anyllm.Provider`. Constructed with a scripted sequence (`WithResponses(...)`) or a matcher-based form (`WhenMessageMatches(re, resp)`). Unexpected inputs fail tests with clear diffs. |
| `EventCollector` | Drains an event channel into a slice with helpers: `ByType`, `ByAgent`, `TokensFor`, `ToolCalls`, `AssertCompleted`. |
| `StubTool` | `NewStubTool(name, schema, fn)` with per-call invocation log. |
| `FakeMCPServer` | In-process MCP server fixture for `mcpadapter` tests. |

No global test state; parallel-safe.

Record/replay from real providers is a v2 concern; v1 relies on
`StubProvider`.

---

## 14. Repo layout

```
sleipnir/                                 # module: sleipnir.dev/sleipnir
├── go.mod  go.sum  LICENSE  README.md  CHANGELOG.md
├── doc.go
├── harness.go  agent.go  event.go  router.go  tool.go
├── tool_typed.go  tool_func.go
├── middleware.go  budget.go  errors.go  config.go
│
├── middleware/
│   ├── compact/
│   ├── accounting/
│   ├── slogobs/
│   └── retry/
│
├── mcpadapter/
├── sleipnirtest/
│
├── examples/
│   ├── hello/
│   ├── orchestrator/
│   └── urd-shaped/
│
└── internal/
    ├── schema/       # invopop/jsonschema wrapper
    └── agentloop/    # loop helpers
```

Single Go module; one version-tag stream; one `CHANGELOG.md`.

Vanity import via Cloudflare Pages at `sleipnir.dev/` serving `go-import`
meta tags that point to `github.com/klaupacius/sleipnir`. Same meta
response required for any subpath (e.g. `sleipnir.dev/sleipnir/mcpadapter`).

---

## 15. Dependencies

| Package | License | Role |
|---|---|---|
| `github.com/mozilla-ai/any-llm-go` | Apache-2.0 | LLM provider boundary |
| `github.com/modelcontextprotocol/go-sdk` | MIT → Apache-2.0 | MCP client (only in `mcpadapter/`) |
| `github.com/invopop/jsonschema` | MIT | Struct-to-JSON-Schema reflection (only in `internal/schema/`) |
| stdlib `log/slog`, `context`, `encoding/json`, `sync`, `time`, `errors` | BSD-3 | Core |

Outbound license: **Apache-2.0**. Transitive license verification via
`go-licenses` in CI.

---

## 16. Non-goals for v1

Deliberately out of scope — listed so contributors don't file issues for
features that were considered and deferred:

- **Declarative agents (YAML/JSON).** `AgentSpec` struct is the only
  authoring path. YAML loader lands only with concrete demand post-v1.
- **Recursive sub-agent dispatch protocols.** Sub-agents are tools. Period.
- **Persistence.** No session store, no memory store, no storage
  integrations. Caller owns history.
- **Retrieval / embeddings / vector stores.**
- **MCP server mode.** Sleipnir consumes MCP via `mcpadapter`; exposing a
  harness *as* an MCP server is a v2 idea.
- **Record/replay testing.** v1 ships `StubProvider`; HTTP-level
  record/replay can come later.
- **OpenTelemetry.** v1 is `slog`-only; `middleware/otelobs` is a v2
  subpackage.
- **Capabilities-based model routing.** v1 router is explicit name-keyed.
- **Streaming partial tool results.** Tool results are single-shot;
  within-tool streaming is v2.
- **Structured HITL (multi-choice, forms).** v1 is text-only.
- **Go `plugin`-style loadable agents.** "Pluggable" means
  import-and-register.

---

## 17. Items deferred to implementation

Small decisions not worth blocking the design doc on; resolved during
implementation:

- Exact `slog` attribute schema used by `SlogObserver`.
- Precise compaction prompt template (tuned during Urd integration).
- Whether `AgentAsTool` accepts an override `Name`/`Description` for the
  parent-facing tool (likely yes).
- Error-wrapping semantics for middleware panics (recover-and-log vs
  propagate-and-fail; leaning propagate).

---

## 18. Milestones (rough)

*Detailed implementation plan produced separately by the writing-plans
skill.*

1. Module + config + `Harness.New` + canonical loop against `StubProvider`
   (no tools, no sub-agents).
2. Tools (`Tool` interface, `NewTypedTool`, `NewFuncTool`, collision
   validation).
3. Parallel tool dispatch.
4. Sub-agents (`AgentAsTool`, isolated history, tagged events).
5. Middleware pipeline + shipped defaults (`retry`, `slogobs`,
   `accounting`, `compact`).
6. HITL + `todo_write`/`todo_read`.
7. `mcpadapter` package.
8. `sleipnirtest` helpers beyond `StubProvider`.
9. Examples + CI + `go-licenses` + release process + `sleipnir.dev`
   vanity setup.
10. Urd integration: port `urd-agent` from Pydantic AI to Sleipnir;
    validate the harness against real SaaS load.
