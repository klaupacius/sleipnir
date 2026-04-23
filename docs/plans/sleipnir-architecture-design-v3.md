# Sleipnir Architecture Design

**Date:** 2026-04-22
**Revised:** 2026-04-22 (post-director-review v3)
**Status:** Approved — ready for implementation
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
│  events stream out via Sink (typed, concurrent-safe)        │
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

// Harness is the central orchestration engine. It is safe for concurrent Run
// calls after all RegisterAgent calls have returned.
//
// By default, calling RegisterAgent after the first Run returns
// ErrHarnessFrozen. Set Config.AllowLateRegistration to permit concurrent
// registration (backed by sync.Map with per-Run snapshot at loop entry).
// Callers who do not need late registration get the freeze-latch safety by
// default. Hot-reload without late registration: construct a new Harness,
// atomic-swap in the caller.
type Harness struct { /* unexported */ }

func New(cfg Config) (*Harness, error)
func (h *Harness) RegisterAgent(spec AgentSpec) error
func (h *Harness) Run(ctx context.Context, in RunInput) (RunOutput, error)
func (h *Harness) AgentAsTool(name string) (Tool, error)

// ---- AgentSpec -------------------------------------------------------------

// AgentSpec defines a named agent's configuration. Registered once on the
// Harness via RegisterAgent. Zero-valued MaxIterations and MaxParallelTools
// are resolved to Config defaults at RegisterAgent time; the stored spec
// contains no zeros.
type AgentSpec struct {
    Name             string
    Description      string                           // shown to a parent agent
    InputSchema      map[string]any                   // JSON Schema for inputs
    SystemPrompt     func(in AgentInput) string
    Tools            []Tool                           // stable tools; includes sub-agents-as-tools
    Middlewares      []Middleware                      // agent-level; overrides Config defaults
    MaxIterations    int                              // 0 → Config default (resolved at registration)
    MaxParallelTools int                              // 0 → Config default (resolved at registration)
}

// AgentInput is the argument passed to AgentSpec.SystemPrompt.
type AgentInput struct {
    Prompt  string
    Input   json.RawMessage                          // validated against AgentSpec.InputSchema
    History []anyllm.Message
}

// ---- AgentInfo -------------------------------------------------------------

// AgentInfo identifies the originating agent in middleware and observer calls.
// Embedded in LLMRequest and ToolCall. Middleware must not mutate these fields.
type AgentInfo struct {
    Name       string
    ParentName string                                // "" for top-level agents
    Depth      int                                   // 0 for top-level
    IsSubAgent bool
}

// ---- RunInput / RunOutput --------------------------------------------------
type RunInput struct {
    AgentName         string
    Prompt            string
    Input             any                             // marshalled against InputSchema
    History           []anyllm.Message                // caller-owned
    Router            ModelRouter                     // required
    Events            Sink                            // nil = discard
    HITL              HITLHandler                     // nil = ask_user tool unavailable
    ExtraTools        []Tool                          // per-run tools (HITL, todo, session-scoped)
    InheritExtraTools bool                            // sub-agents inherit ExtraTools (default true; see §5.1)
    MaxTotalTokens    int                             // run-wide cumulative; 0 = unlimited (see §11)
}

type RunOutput struct {
    Text     string                                   // final assistant text (== Messages[len-1].Content)
    Messages []anyllm.Message                         // updated history
    Usage    Usage                                    // aggregated parent + subs
    Stopped  StopReason                               // see StopReason constants
}

// ---- HITLHandler -----------------------------------------------------------

// HITLHandler is the callback for human-in-the-loop interactions. Provided
// via RunInput.HITL. The harness calls AskUser when a tool invocation of
// ask_user is dispatched; the implementation is responsible for routing the
// question to the appropriate human and returning their reply. Concurrent
// calls to AskUser are allowed (parallel ask_user tool_use blocks).
type HITLHandler interface {
    AskUser(ctx context.Context, agent, question, contextBlurb string) (string, error)
}

// ---- ModelRouter -----------------------------------------------------------
type ModelConfig struct {
    Provider        anyllm.Provider
    Model           string
    ReasoningEffort anyllm.ReasoningEffort
    Temperature     *float64
    MaxOutputTokens *int
}

// ModelRouter resolves the model configuration for a given agent. Resolve is
// called once per LLM call (not cached across iterations). Implementations
// may be stateful (e.g., feature-flag routing, tenant-aware routing).
// Middleware that cares about model identity must capture the resolved
// ModelConfig at call time and not assume stability across iterations.
//
// Implementations SHOULD cache unless they have a reason not to. Resolve may
// be called 20+ times per agent per Run (once per iteration), multiplied
// across sub-agents. See CachedRouter for a ready-made caching wrapper.
type ModelRouter interface {
    Resolve(ctx context.Context, agentName string) (ModelConfig, error)
}

type MapRouter struct {
    Default   ModelConfig
    Overrides map[string]ModelConfig                  // keyed by AgentSpec.Name
}

// CachedRouter wraps a ModelRouter and caches resolved ModelConfig values
// for the duration of a single Run, keyed by agent name. Callers who need
// per-iteration resolution (e.g., canary-rollout routers) should not wrap.
type CachedRouter struct { /* unexported; wraps a ModelRouter */ }

func NewCachedRouter(inner ModelRouter) *CachedRouter

// ---- Tool ------------------------------------------------------------------

// Tool is the interface for all tools available to agents. Invoke is called
// by the harness during parallel tool dispatch.
//
// Error semantics:
//   - Return (ToolResult{IsError: true}, nil) for structured, LLM-readable
//     failures where the tool controls the error message content.
//   - Return (ToolResult{}, error) for infrastructure failures that cannot be
//     meaningfully summarized for the LLM. The harness wraps the error as
//     ToolResult{IsError: true, Content: "tool execution failed: " + err.Error()},
//     emits an ErrorEvent, and continues the run (error isolation; §5).
//
// Prefer ToolResult{IsError: true} for LLM-observable failures. Reserve the
// error return for infrastructure problems.
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

// Middleware is a marker interface for middleware values. A middleware may
// implement any subset of ContextRewriter, LLMObserver, ToolObserver,
// and RetryPolicy. The harness discovers capabilities via type assertion.
type Middleware interface{ middleware() }

// Helpers:

// NewTypedTool creates a Tool whose input schema is reflected from the struct
// type T at construction time. Returns an error if schema reflection fails
// (e.g., cycles, unsupported types in T).
func NewTypedTool[T any](name, desc string,
    fn func(context.Context, T) (ToolResult, error)) (Tool, error)

func NewFuncTool(name, desc string, schema map[string]any,
    fn func(context.Context, json.RawMessage) (ToolResult, error)) Tool

// ---- Built-in tool constructors --------------------------------------------

// AskUserTool returns a Tool that, when invoked by the LLM, delegates to the
// provided HITLHandler. The tool has a fixed schema:
//   {"question": "string", "context": "string (optional)"}
// Typically passed via RunInput.ExtraTools. Emits a data-only QuestionEvent
// *before* calling the handler, so observers see the question regardless of
// whether the handler returns.
func AskUserTool(handler HITLHandler) Tool

// TodoReadTool and TodoWriteTool return tools for agent-scoped planning lists
// with full-replace semantics. State is scoped per-Run, keyed by ToolCallID
// (see §5.2, §10). Typically passed via RunInput.ExtraTools.
func TodoReadTool() Tool
func TodoWriteTool() Tool

// ---- StopReason ------------------------------------------------------------

type StopReason string

const (
    StopDone            StopReason = "done"
    StopIterationBudget StopReason = "iteration_budget"
    StopTokenBudget     StopReason = "token_budget"
    StopHITLTimeout     StopReason = "hitl_timeout"
    StopHITLCancelled   StopReason = "hitl_cancelled"
)

// ---- BufferedSink ----------------------------------------------------------

// BufferedSink is a Sink implementation that buffers up to a configured number
// of events internally. If the buffer is full, Send drops the event
// (non-blocking). Safe for concurrent use. Use DroppedCount to monitor drops.
type BufferedSink struct { /* unexported */ }

func NewBufferedSink(ctx context.Context, size int) *BufferedSink
func (s *BufferedSink) Send(e Event)
func (s *BufferedSink) Events() <-chan Event
func (s *BufferedSink) DroppedCount() int64
```

---

## 5. Canonical agent loop

Implemented once in `harness.go`. Every agent — parent or sub-agent —
executes the same loop.

```text
effective := resolvedSpec(spec)   // zeros replaced at RegisterAgent time
allTools := append(effective.Tools, runInput.ExtraTools...)
validateToolNames(allTools)       // ErrToolNameCollision on duplicates

for i := 0; i < effective.MaxIterations; i++ {
    modelCfg, err := router.Resolve(ctx, effective.Name)
    if err != nil { return err }

    params := assembleParams(effective, history, modelCfg)
    params.Agent = AgentInfo{Name: effective.Name, ParentName: parentName,
                             Depth: depth, IsSubAgent: depth > 0}
    applyBefore(middlewares, ctx, &params)

    resp, err := providerCompletionWithRetry(ctx, params)
    applyAfter(middlewares, ctx, &params, &resp, err)
    if err != nil { return err }

    history = append(history, resp.Choices[0].Message)
    runState.addTokens(resp.Usage)     // sole call-site writer; concurrent across goroutines
    emitTokenAndToolCallEvents(resp, sink)

    if runState.overBudget() {
        return RunOutput{..., Stopped: StopTokenBudget}
    }

    toolCalls := resp.Choices[0].Message.ToolCalls
    if len(toolCalls) == 0 {
        return RunOutput{Text: resp.Choices[0].Message.Content,
                         Messages: history,
                         Usage: accum.Usage(),
                         Stopped: StopDone}
    }

    results := dispatchInParallel(ctx, toolCalls, allTools, h, runState,
                                  sink, effective.MaxParallelTools)
    history = append(history, toolResultMessages(results)...)
}
return ErrIterationBudget
```

Dispatch rules:

- Each `ToolCall` is resolved against `allTools` (`spec.Tools` +
  `RunInput.ExtraTools`) by name. If the matched `Tool` is a sub-agent
  wrapper (from `AgentAsTool`; detected via unexported `subAgentTool`
  marker interface), the harness recurses into `Run` with a fresh
  `RunInput` (new goroutine, isolated history, parent-tagged events).
  Otherwise, `Tool.Invoke` runs in a goroutine.
- Parallelism bounded by `MaxParallelTools` via a semaphore.
- Errors from one tool do **not** cancel siblings; they return as
  `IsError: true` tool_results so the LLM can react.
- `context.Context` cancellation cascades: cancelling a Run ctx cancels all
  in-flight LLM calls, tools, and sub-agents cooperatively.

### 5.1. Sub-agent `RunInput` inheritance

When the harness dispatches a sub-agent tool, it constructs a child
`RunInput`. Inheritance rules:

| Field | Inherited? | Notes |
|---|---|---|
| `AgentName` | No | Set to sub-agent's own name |
| `Prompt` / `Input` | No | From parent LLM's `tool_use` args, validated against sub-agent `InputSchema` |
| `History` | No | Starts empty (strict isolation, §2.4) |
| `Router` | Yes | Same router instance; sub-agent resolved by its own name |
| `Events` | Yes | Same sink; events tagged by sub-agent's `AgentName` |
| `HITL` | Yes | Same handler; calls tagged with sub-agent name |
| `ExtraTools` | Conditional | Inherited if parent's `InheritExtraTools` is true (default). See security note below. |
| `InheritExtraTools` | Yes | Propagated to sub-agents so the policy applies recursively |
| `MaxTotalTokens` | Shared | Via `runState` budget tracker (§5.2); not a separate field |
| Deadline | Via `ctx` | Sub-agents inherit the parent's `context.Context` |
| Middlewares | Yes | Effective middleware chain (agent-level overrides config-level) |

**Security note on `ExtraTools` inheritance:** When `InheritExtraTools` is
true (default), all sub-agents receive the parent's `ExtraTools`, including
`AskUserTool`. This means a sub-agent invoked with a prompt-injected input
could call `ask_user` and present misleading questions to the human.
Mitigation lives in the `HITLHandler` implementation (validate, scope, or
rate-limit by agent name). Callers who need per-sub-agent tool scoping
should set `InheritExtraTools: false` and provide tools explicitly via each
sub-agent's own `spec.Tools` or via a custom `AgentAsTool` wrapper.

### 5.2. `runState` — per-Run shared state

A `runState` struct is created at the start of each top-level `Run` call
and threaded through recursive sub-agent `Run` calls as a direct argument
on an internal `runLoop` function (not via `context.Value`).

`runState` is always passed and stored as `*runState` (pointer). It must
never be copied by value — `go vet`'s `copylocks` check is enabled in CI
to enforce this.

```go
type runState struct {
    // Token budget enforcement. The agent loop calls addTokens() after each
    // LLM response. Multiple goroutines (parent loop + parallel sub-agent
    // loops) call addTokens() concurrently on the same *runState, so an
    // atomic is the correct primitive — each call site is the sole writer
    // in its goroutine, but writes are concurrent across goroutines.
    // TokenAccountant middleware is a pure observer — it never touches this
    // counter. See "Design rationale: atomic vs mutex" in Appendix B.
    tokensUsed  atomic.Int64
    tokenBudget int64         // from RunInput.MaxTotalTokens; 0 = unlimited

    // Per-invocation todo lists, keyed by ToolCallID (unique per dispatch).
    // Top-level Run uses a synthetic root ID. Each sub-agent invocation gets
    // its own list. Lists persist across iterations within a single Run but
    // are not surfaced in RunOutput (callers reconstruct from TodoEvent).
    todos sync.Map            // map[string][]TodoItem  (key = ToolCallID)

    // Compaction watermarks, keyed by agent name. See §7.1 (compact middleware).
    compactState sync.Map     // map[string]*compactWatermark
}
```

**Invariant (tested in CI):** `TokenAccountant`'s aggregate total must
equal `runState.tokensUsed` at the end of every `Run`. A divergence
indicates double-counting or a missed observation. Tested via
`sleipnirtest.AssertTokenInvariant`. This is a CI-time guarantee, not a
production assertion.

**Budget enforcement:**
- The loop calls `runState.addTokens()` after each LLM response. Writes
  are concurrent across goroutines (parent + sub-agents) — hence `atomic.Int64`.
- `TokenAccountant` is a pure `LLMObserver`. It reads `LLMResponse.Usage`
  and aggregates per-agent into its own map. **It never touches the budget
  counter.** Budget enforcement is a harness correctness concern, not a
  middleware concern. Middleware is swappable; the budget check is not.
- Check happens after each LLM response. If over budget, current response
  is kept, next iteration returns `ErrTokenBudget`.
- Sub-agent budget overrun: sub-agent `Run` returns `ErrTokenBudget`,
  harness wraps as `ToolResult{IsError: true}` to the parent LLM so it
  can adapt (matches error-isolation policy).

---

## 6. Events and streaming

```go
// Sink receives events from the harness. Implementations must be safe for
// concurrent use from multiple goroutines. Send is non-blocking by contract;
// the harness does not mandate a delivery policy. Implementations own their
// buffering and drop strategy. Callers who need guaranteed delivery should
// supply their own Sink (e.g., a bounded channel with a blocking send and
// caller-owned timeout). See BufferedSink for a ready-made drop-on-full
// implementation with a DroppedCount() metric.
type Sink interface { Send(Event) }

type Event interface{ eventMarker() }

type AgentStartEvent  struct { AgentName, ParentName string }
type AgentEndEvent    struct { AgentName string; Usage Usage; Stopped StopReason }
type TokenEvent       struct { AgentName, Text string }
type ThinkingEvent    struct { AgentName, Text string }
type ToolCallEvent    struct { AgentName, ToolCallID, ToolName string; Args json.RawMessage }
type ToolResultEvent  struct { AgentName, ToolCallID string; Result string; IsError bool }
type QuestionEvent    struct { AgentName string; QuestionID string; Question string }
type TodoEvent        struct { AgentName string; ToolCallID string; Todos []TodoItem }
type ErrorEvent       struct { AgentName string; Err error }
```

### Semantics

- `Send` MUST be safe for concurrent use.
- `Send` is non-blocking by contract. The harness does not mandate a
  delivery policy — implementations choose their own buffering/drop/block
  strategy. `BufferedSink` drops on full and exposes `DroppedCount()`.
  Callers who need reliable delivery supply their own `Sink`.
- **No cross-agent ordering guarantee.** Per-agent events are emitted in
  loop order.
- `AgentEndEvent` for a sub-agent is emitted *before* the corresponding
  `ToolResultEvent` on the parent stream.
- Sub-agent events flow through the **same** sink, tagged by `AgentName`
  (and `ParentName` on start).
- Consumer filters/rolls up; harness stays policy-free.
- `QuestionEvent` is data-only. The actual reply flow is handled by
  `HITLHandler` (§4). The event is emitted *before* calling the handler so
  observers see the question regardless of handler outcome. A dropped
  `QuestionEvent` means an observer misses a log line, not that the user's
  question vanishes — the reply path is `HITLHandler.AskUser`, which is
  independent of the event stream.
- `TodoEvent` carries the `ToolCallID` that scopes the todo list (§10).
- Urd's API layer maps each `Event` variant to a `text/event-stream` payload
  matching the existing SSE contract (`token`, `thinking`, `tool_call`,
  `tool_done`, `question`, `done`, `error`).

---

## 7. Middleware

A middleware value implements the `Middleware` marker interface and may
additionally implement any subset of the interfaces below. The harness
discovers capabilities via type assertion and calls what's implemented.
Order is preserved from registration.

Middleware is configured at two levels: `Config.Middlewares` provides
application-wide defaults; `AgentSpec.Middlewares` overrides per-agent.
Agent-level takes precedence.

All middleware interfaces receive agent identity via `AgentInfo` embedded
in `LLMRequest` and `ToolCall`. Middleware must not mutate identity fields.

```go
// ContextRewriter may mutate fields on *LLMRequest (including Messages) to
// transform the request before the LLM call. The compactor uses this to
// replace a prefix of messages with a summary.
type ContextRewriter interface {
    Middleware
    RewriteBeforeLLMCall(ctx context.Context, req *LLMRequest) error
}
type LLMObserver interface {
    Middleware
    OnLLMCall(ctx context.Context, req *LLMRequest, resp *LLMResponse, err error)
}
type ToolObserver interface {
    Middleware
    OnToolCall(ctx context.Context, call *ToolCall, result *ToolResult, err error)
}
type RetryPolicy interface {
    Middleware
    ShouldRetry(ctx context.Context, attempt int, err error) (retry bool, backoff time.Duration)
}
```

`LLMRequest` and `ToolCall` both embed `AgentInfo` (§4), providing
`Name`, `ParentName`, `Depth`, and `IsSubAgent` to every middleware call.

### Shipped middlewares (subpackages)

| Subpackage | Provides | Role |
|---|---|---|
| `middleware/compact` | `Compactor` (`ContextRewriter`) | LLM-based summarization of oldest messages when prompt exceeds `CompactThreshold` (default 75% of model context). Summarizer model from `CompactModel` (default `claude-haiku-4-5`). Off for sub-agents (uses `AgentInfo.IsSubAgent`). Uses a stable compaction watermark (§7.1) to avoid re-summarizing. On failure: proceed uncompacted, emit `ErrorEvent`, let `anyllm.ErrContextLength` bubble up normally if the LLM call fails. Mutates `LLMRequest.Messages` via pointer (the intended mechanism for `ContextRewriter`). |
| `middleware/accounting` | `TokenAccountant` (`LLMObserver`) | Pure observer: reads `LLMResponse.Usage` and aggregates per-agent into its own map. **Never touches the budget counter in `runState`.** Surfaced via `RunOutput.Usage`. Must equal `runState.tokensUsed` at run end (CI-tested invariant via `sleipnirtest.AssertTokenInvariant`). |
| `middleware/slogobs` | `SlogObserver` (`LLMObserver` + `ToolObserver`) | Structured logs with `AgentInfo` attributes. Level + format from env. |
| `middleware/retry` | `DefaultRetryPolicy` (`RetryPolicy`) | LLM-call retries: exponential backoff + jitter, max 3 attempts (`SLEIPNIR_MAX_LLM_RETRIES`). Retries `anyllm.ErrRateLimit` and `anyllm.ErrProvider`. Does *not* retry `ErrInvalidRequest`, `ErrContextLength`, `ErrAuthentication`, `ErrContentFilter`, `ErrModelNotFound`, `ErrInsufficientFunds`. Tool errors are never retried by the harness. Uses the `ModelConfig` resolved for the current call, not a cached value. |

OpenTelemetry observer arrives as `middleware/otelobs` in v2.

### 7.1. Compaction watermark

The compactor tracks a stable compaction marker per-agent in
`runState.compactState`, keyed by agent name. Sub-agents get their own
watermark (their histories are isolated).

The watermark is **not** a raw slice index into the mutable message
history (indices shift when a prefix is replaced by a summary message).
Instead, the compactor stores the count of *original* messages that have
been compacted. On each turn, the compactor walks from that offset,
skipping already-summarized content. This avoids re-summarizing summaries
and is robust to synthetic messages injected by other middleware.

---

## 8. Tools

### Authoring

- `NewTypedTool[T](name, desc, fn)` — input struct `T` with `json:` and
  `jsonschema:` tags; schema reflected at construction time via
  [`invopop/jsonschema`](https://github.com/invopop/jsonschema). Returns
  `(Tool, error)` because schema reflection can fail (cycles, unsupported
  types). Primary path for Go-native tools.
- `NewFuncTool(name, desc, schema, fn)` — raw `json.RawMessage` in,
  `ToolResult` out, hand-written schema. Escape hatch for dynamic /
  union schemas.

### Built-in tools

Built-in tools (`AskUserTool`, `TodoReadTool`, `TodoWriteTool`) are
ordinary `Tool` values constructed by the caller and passed via
`RunInput.ExtraTools`. They are not special-cased in `AgentSpec`. This
keeps the `AgentSpec` struct stable as new built-ins are added and
generalizes to any session-scoped tool (auth-scoped, tenant-scoped, etc.).

```go
h.Run(ctx, sleipnir.RunInput{
    AgentName: "researcher",
    ExtraTools: []sleipnir.Tool{
        sleipnir.AskUserTool(myHITLHandler),
        sleipnir.TodoWriteTool(),
        sleipnir.TodoReadTool(),
    },
    InheritExtraTools: true,  // sub-agents also get these tools (default)
    // ...
})
```

Todo tools close over per-`Run` state managed internally by the harness
(via `runState.todos`, keyed by `ToolCallID`). `AskUserTool` closes over
the provided `HITLHandler`.

### MCP adapter

Separate package `sleipnir.dev/sleipnir/mcpadapter`. Depends on
`github.com/modelcontextprotocol/go-sdk`. Core does not depend on MCP.

```go
tools, err := mcpadapter.LoadTools(ctx, mcpClient,
    mcpadapter.WithPrefix("urd_"),   // optional; e.g. urd_search_fts
)
// Pass via ExtraTools or spec.Tools as appropriate:
spec.Tools = append(spec.Tools, tools...)
```

- Wraps each MCP tool into a Sleipnir `Tool`; `Invoke` forwards to MCP
  `tools/call`.
- Caller owns the MCP client lifecycle.
- `LoadTools` snapshots the tool list at call time. MCP tool-change
  notifications are not subscribed to. Consumers must re-call `LoadTools`
  to pick up server-side changes. Live subscription is a v2 concern.

### Name collisions

Tool name uniqueness is validated at two points:

1. `RegisterAgent` validates `AgentSpec.Tools` at registration time.
2. At `Run` start, `spec.Tools` + `RunInput.ExtraTools` are validated
   together.

Duplicate names → hard error (`ErrToolNameCollision`). Namespacing is the
caller's responsibility via `WithPrefix`; the harness does not silently
rename.

### Sub-agent tool detection

`AgentAsTool` returns a concrete type that implements an unexported marker
interface:

```go
type subAgentTool interface {
    Tool
    subAgentName() string
}
```

The harness type-asserts during dispatch. The public `Tool` interface
stays minimal.

---

## 9. Human-in-the-loop (`ask_user`)

HITL is implemented as an ordinary tool (`AskUserTool`) backed by a
caller-provided `HITLHandler` interface (§4). The tool has a fixed schema:

```json
{"question": "string", "context": "string (optional)"}
```

Execution:

1. LLM emits a `tool_use` block calling `ask_user`.
2. Harness emits a data-only `QuestionEvent{AgentName, QuestionID, Question}`
   *before* calling the handler. Observers see the question regardless of
   handler outcome.
3. Harness calls `HITLHandler.AskUser(ctx, agent, question, context)`.
4. Handler blocks until user reply, `ctx.Done()`, or
   `SLEIPNIR_HITL_TIMEOUT` (default 30 min).
5. On reply → string becomes `tool_result` content; loop continues.
6. On cancel / timeout → `Run` returns `ErrHITLCancelled` /
   `ErrHITLTimeout`.

Parallel `ask_user` is allowed. If the LLM emits two `tool_use` blocks
both calling `ask_user` in one turn, the harness calls `AskUser`
concurrently from two goroutines. The handler implementation decides how
to multiplex (e.g., Urd serializes in the UI).

Sub-agents inherit the `HITLHandler` from `RunInput` when
`InheritExtraTools` is true (§5.1).
`QuestionEvent.AgentName` lets the consumer route replies correctly.

---

## 10. Todo / planning tool

Provided as two built-in tool constructors (`TodoWriteTool`,
`TodoReadTool`) with **full-replace** semantics:

```text
todo_write: {tasks: [{id: string, text: string,
                      status: "pending"|"in_progress"|"done"}]}
    replaces the current invocation's todo list.
todo_read:  {}
    returns the current invocation's todo list.
```

- State lives in `runState.todos` (§5.2), keyed by **`ToolCallID`**
  (unique per tool dispatch). The top-level `Run` uses a synthetic root
  ID. Each sub-agent invocation gets its own list, eliminating races
  between parallel sub-agents that share an agent name.
- The list persists across iterations within one `Run` invocation
  (cross-turn planning) but is not cleared between turns.
- On every update, `TodoEvent{AgentName, ToolCallID, Todos}` is emitted.
- **Not persisted by the harness.** Not included in `RunOutput`. Callers
  who want session-scoped todos observe `TodoEvent` and write to their own
  store. If the event-accumulation boilerplate becomes painful, a
  `sleipnirtest.TodoCollector` helper can be added — but todo state stays
  out of `RunOutput` per principle §2.6 ("harness owns no data").

Full-replace (vs. granular `todo_update_status` / `todo_remove`) was chosen
for: lower prompt overhead, natural idempotence, fewer LLM reasoning paths,
fewer harness edge cases. Validated pattern from Claude Code's `TodoWrite`.

---

## 11. Budgets, limits, cancellation

| Knob | Where | Default |
|---|---|---|
| `MaxIterations` | `AgentSpec` (per agent, resolved at registration) | 20 (Config) |
| `MaxParallelTools` | `AgentSpec` (per agent, resolved at registration) | 8 (Config) |
| `MaxTotalTokens` | `RunInput` (per run, parent + subs via `runState`) | 0 (unlimited) |

`MaxTotalTokens` is a **cumulative** budget across the parent agent and all
sub-agents. Example: a 100k-token budget with 5 sub-agents each consuming
30k tokens → the run halts when cumulative usage hits 100k, regardless of
which agent consumed it. There is no per-agent token budget in v1; callers
who need per-agent limits should implement them via a custom `ContextRewriter`
middleware or by checking `RunOutput.Usage` per-agent breakdowns.

Deadlines are set via `context.Context` by the caller. No separate
`Deadline` field on `RunInput`.

Typed errors: `ErrIterationBudget`, `ErrTokenBudget`,
`ErrHITLTimeout`, `ErrHITLCancelled`, `ErrCompactionFailed`,
`ErrAgentNotRegistered`, `ErrToolNameCollision`, `ErrHarnessFrozen`.

`context.Context` cancellation cascades. Sub-agents always inherit the
parent's ctx.

---

## 12. Configuration (twelve-factor)

```go
type Config struct {
    DefaultMaxIterations    int
    DefaultMaxParallelTools int
    CompactThreshold        float64        // 0.75 = 75% of model context; validated (0.0, 1.0]
    CompactModel            string
    MaxLLMRetries           int
    HITLTimeout             time.Duration
    LogLevel                slog.Level
    LogFormat               string         // "json" | "text"
    Middlewares             []Middleware    // application-wide defaults; overridden by AgentSpec.Middlewares
    AllowLateRegistration   bool           // permit RegisterAgent after first Run (default false)
}

func LoadConfigFromEnv() (Config, error)
```

Environment variables read by `LoadConfigFromEnv`:

```text
SLEIPNIR_MAX_ITERATIONS
SLEIPNIR_MAX_PARALLEL_TOOLS
SLEIPNIR_COMPACT_THRESHOLD          # e.g. 0.75
SLEIPNIR_COMPACT_MODEL
SLEIPNIR_MAX_LLM_RETRIES
SLEIPNIR_HITL_TIMEOUT
SLEIPNIR_LOG_LEVEL
SLEIPNIR_LOG_FORMAT
SLEIPNIR_ALLOW_LATE_REGISTRATION    # "true" | "false"
```

Missing vars fall back to documented defaults. No file reads. No globals.
Both `Config{}` (tests) and `LoadConfigFromEnv()` (production) are
first-class.

---

## 13. Testing support (`sleipnirtest/`)

| Helper | Purpose |
|---|---|
| `StubProvider` | Implements `anyllm.Provider`. Constructed with a scripted sequence (`WithResponses(...)`) or a matcher-based form (`WhenMessageMatches(re, resp)`). Unexpected inputs fail tests with clear diffs. |
| `EventCollector` | Drains a `Sink` into a slice with helpers: `ByType`, `ByAgent`, `TokensFor`, `ToolCalls`, `AssertCompleted`. |
| `StubTool` | `NewStubTool(name, schema, fn)` with per-call invocation log. |
| `FakeMCPServer` | In-process MCP server fixture for `mcpadapter` tests. |
| `TodoCollector` | Drains `TodoEvent`s and exposes final per-invocation todo state. Convenience for tests and callers who need session-scoped todos. |
| `AssertTokenInvariant` | Asserts `TokenAccountant` aggregate equals `runState.tokensUsed` at run end. CI-time only. |

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
├── tool_typed.go  tool_func.go  tool_builtin.go
├── middleware.go  budget.go  errors.go  config.go
├── sink.go  runstate.go  cached_router.go
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
├── vanity/                               # go-import meta HTML (version-controlled)
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
meta tags that point to `github.com/klaupacius/sleipnir`. Meta HTML is
version-controlled under `vanity/` for reproducibility. Same meta response
required for any subpath (e.g. `sleipnir.dev/sleipnir/mcpadapter`).

---

## 15. Dependencies

| Package | License | Role |
|---|---|---|
| `github.com/mozilla-ai/any-llm-go` | Apache-2.0 | LLM provider boundary |
| `github.com/modelcontextprotocol/go-sdk` | MIT | MCP client (only in `mcpadapter/`) |
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
- **Mid-turn tool cancellation.** `context.Context` cancellation is the
  only mechanism. The LLM cannot cancel a previously-dispatched tool.
- **MCP live tool-list subscription.** `LoadTools` snapshots; consumers
  re-call to refresh.
- **Per-agent token budgets.** v1 has cumulative-only `MaxTotalTokens`.
- **Structured sub-agent results.** `ToolResult.StructuredData` is a v2
  concern; v1 sub-agents return text only.

---

## 17. Items deferred to implementation

Small decisions not worth blocking the design doc on; resolved during
implementation:

- Exact `slog` attribute schema used by `SlogObserver`.
- Precise compaction prompt template (tuned during Urd integration).
- Whether `AgentAsTool` accepts an override `Name`/`Description` for the
  parent-facing tool (likely yes).
- Error-wrapping semantics for middleware panics (propagate-and-fail;
  panic is reserved for genuine programmer errors like nil harness).
- Verify `anyllm.Message` type exists in the actual `any-llm-go`
  dependency before first milestone PR.

---

## 18. Milestones (rough)

*Detailed implementation plan produced separately by the writing-plans
skill.*

1. Module + config + `Harness.New` + freeze latch + canonical loop against
   `StubProvider` (no tools, no sub-agents). CI with `go vet` copylocks.
2. Tools (`Tool` interface, `NewTypedTool`, `NewFuncTool`, collision
   validation).
3. Parallel tool dispatch + `ExtraTools` merging + `InheritExtraTools`.
4. Sub-agents (`AgentAsTool`, isolated history, tagged events, inheritance
   table).
5. `runState` + token budget enforcement + `AssertTokenInvariant` test.
6. Middleware pipeline + shipped defaults (`retry`, `slogobs`,
   `accounting`, `compact` with stable watermark).
7. `HITLHandler` + `AskUserTool` + `QuestionEvent` (data-only).
8. `TodoWriteTool` / `TodoReadTool` + `TodoEvent` (keyed by `ToolCallID`).
9. `CachedRouter` wrapper.
10. `mcpadapter` package (snapshot semantics).
11. `sleipnirtest` helpers (`EventCollector`, `TodoCollector`, `StubTool`,
    `FakeMCPServer`).
12. Examples + CI + `go-licenses` + release process + `sleipnir.dev`
    vanity setup.
13. Urd integration: port `urd-agent` from Pydantic AI to Sleipnir;
    validate the harness against real SaaS load.

---

## Appendix A. Review changelog

### v1 → v2 (architecture review)

| ID | Change | Source |
|---|---|---|
| M1 | Added `AgentInfo` struct; embedded in `LLMRequest` and `ToolCall` for middleware identity | Review |
| M2 | Replaced `QuestionEvent.ReplyCh` with `HITLHandler` interface; `QuestionEvent` is data-only | Review |
| M3 | Specified `Sink` concurrency, blocking, and ordering semantics | Review |
| M4 | Added §5.1 sub-agent inheritance table | Review |
| M5 | Added §5.2 `runState` with token budget mechanism; one-writer rule; accounting invariant test | Review |
| M6 | Dropped `EnableAskUser`/`EnableTodoList` bools; added `RunInput.ExtraTools` and built-in tool constructors | Review |
| M7 | Defined "agent instance" = per-`Run`; todo state in `runState` keyed by agent name | Review |
| M8 | Documented `Tool.Invoke` error semantics (error vs IsError) in godoc | Review |
| M9 | Added concurrency model statement; `ErrHarnessFrozen` on post-freeze `RegisterAgent` | Review |
| M10 | Defaults resolved at `RegisterAgent` time; pseudocode updated | Review |
| m1 | Removed `Deadline` from `RunInput`; callers use `context.WithDeadline` | Review |
| m2 | `CompactThreshold` changed to `float64` (0.75) | Review |
| m3 | Defined `AgentInput` type | Review |
| m4 | `ModelRouter.Resolve` takes `ctx` and returns `error`; documented stateful routers | Review |
| m5 | Documented sub-agent tool detection via unexported `subAgentTool` interface | Review |
| m6 | Enumerated `StopReason` constants | Review |
| m7 | Specified compaction failure policy (proceed uncompacted, emit `ErrorEvent`) | Review |
| m8 | `NewTypedTool[T]` returns `(Tool, error)` | Review |
| m9 | Compaction watermark uses stable original-message count, not raw slice index | Review |
| m10 | MCP adapter: documented snapshot-once semantics | Review |
| — | `Middlewares` on `AgentSpec` (agent-level overrides config defaults) | Review |
| — | `Middleware` marker interface replaces `[]any` | Review |
| — | Vanity import HTML version-controlled under `vanity/` | Review |
| — | `Sink` renamed from `EventSink`; `BufferedSink` in main package | Review |
| — | `RunOutput.Text` documented as convenience accessor | Review |
| — | go-sdk license confirmed as MIT (dropped "→ Apache-2.0") | Review |

### v2 → v3 (director review)

| ID | Change | Source | Action |
|---|---|---|---|
| D3 | `Config.AllowLateRegistration` for SaaS hot-reload | Director | Accepted |
| D4 | `CachedRouter` wrapper; `ModelRouter` godoc caching guidance | Director | Accepted |
| D5 | `RunInput.InheritExtraTools bool` + security note in §5.1 | Director | Accepted (simplified) |
| D6 | Todo state keyed by `ToolCallID` (not agent name) to eliminate race | Director | Accepted |
| D10 | Cumulative budget example in §11; per-agent budgets deferred to v2 | Director | Accepted (doc-only) |
| D-sink | `BufferedSink` returns `*BufferedSink` with `Events()` and `DroppedCount()` | Director | Accepted |
| D-godoc | `NewTypedTool` godoc explains `error` return | Director | Accepted |
| D-vet | CI `go vet copylocks` check for `runState` | Director | Accepted |
| D-verify | Verify `anyllm.Message` exists in dependency | Director | Accepted (deferred to impl) |

### v2 → v3: items explicitly retained (with rationale)

These items were challenged in the director review and retained after
analysis. Listed here so future reviewers do not re-litigate.

| Item | Director's concern | Rationale for retaining |
|---|---|---|
| `atomic.Int64` on `tokensUsed` | "Sole writer, use mutex" | `runState` is shared `*runState` across the parent loop and every parallel sub-agent goroutine. Writes to `tokensUsed` are genuinely concurrent. Atomic is correct; a mutex would either be held across the LLM call (deadlock risk) or acquired/released for a counter bump (equivalent to atomic but slower). `go vet copylocks` in CI catches accidental value copies. |
| `TokenAccountant` as observer, not source of truth | "Make accountant the only writer, remove `runState.tokensUsed`" | Budget enforcement is a harness correctness concern. Middleware is swappable; the budget check is not. Coupling a non-negotiable correctness check to a swappable middleware creates silent failure modes (replace the accountant → budget enforcement disappears). The invariant test (`AssertTokenInvariant`) proves the two counters agree in CI. |
| `ContextRewriter` mutation via `*LLMRequest` pointer | "Can't modify the request meaningfully" | `req *LLMRequest` is a pointer — mutation is the intended mechanism. The compactor modifies `req.Messages` directly. No side channel. |
| `SystemPrompt func(AgentInput) string` (no parallel `SystemPromptString` field) | "Function call for static prompts is wasteful" | Function-call overhead is noise against an LLM round-trip. A static prompt is `func(AgentInput) string { return "..." }` — two lines. Adding a parallel field doubles API surface for a non-problem. |
| `sleipnirtest` package name | "`sleipnir/test` would be more Go-like" | `xxxtest` is the standard Go pattern: `httptest`, `iotest`, `fstest`, `synctest`. |
| `Sink.Send` non-blocking contract | "Non-blocking + drop is unreliable" | `QuestionEvent` is informational-only; the reply path is `HITLHandler.AskUser`, independent of the event stream. The harness does not mandate delivery policy; callers who need guaranteed delivery supply their own `Sink` with blocking semantics. `BufferedSink.DroppedCount()` surfaces drop metrics. |

---

## Appendix B. Design rationale: `atomic.Int64` vs `sync.Mutex` for `tokensUsed`

The `runState.tokensUsed` counter receives concurrent writes from multiple
goroutines: the parent agent's loop and every parallel sub-agent's loop all
call `runState.addTokens(...)` on the **same** `*runState`.

```text
Parent loop goroutine:
    resp := llmCall(...)
    runState.addTokens(resp.Usage.TotalTokens)  // write

Sub-agent goroutine A:                    Sub-agent goroutine B:
    resp := llmCall(...)                      resp := llmCall(...)
    runState.addTokens(resp.Usage...)         runState.addTokens(resp.Usage...)
    // write, concurrent with parent          // write, concurrent with A and parent
```

`atomic.Int64.Add()` is the correct primitive for a single shared counter
with concurrent writers. A `sync.Mutex` alternative would either:

1. Be held across the LLM call (blocking all other agents from updating
   the counter — effectively serializing budget checks), or
2. Be acquired and released around the single `Add` operation — which is
   semantically identical to an atomic but with higher overhead.

`runState` is always passed as `*runState`. CI enforces `go vet -copylocks`
to catch accidental value copies that would break the atomic.
