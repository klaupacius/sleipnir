# Re: Director's Review of Sleipnir v2

Jo — the director's feedback landed. Overall he's positive ("ready for implementation with critical fixes," "exceptional work") and most of his concerns are worth addressing. But several of his "critical" items are wrong or half-wrong, and I don't want you quietly capitulating to seniority when the design is right. Here's my read, point by point, with what to do.

---

## Where the director is wrong (push back, with evidence)

### Critical #1 — `atomic.Int64` for `runState.tokensUsed`

**He's wrong.** His premise — "sole writer, so use a mutex" — misreads the concurrency model.

The loop is the sole writer *at each call site*, but there are **multiple concurrent call sites**: the parent agent's loop and every parallel sub-agent's loop all call `runState.addTokens(...)` on the **same** `*runState` from different goroutines. That's genuinely concurrent writes. `atomic.Int64` is the correct primitive.

His secondary concern ("copying `runState` via argument passing shouldn't happen with atomics") is also wrong — `runState` is threaded as `*runState`, never by value. `go vet`'s `copylocks` check will catch any accidental copy.

**Response to the director:** "`runState` is shared by pointer across the parent loop and every parallel sub-agent goroutine. Writes to `tokensUsed` are concurrent by construction. Atomic is correct; a mutex would either be held across the LLM call (bad) or acquired/released around a counter bump (equivalent to atomic but slower)."

Do add a `go vet` check in CI to catch accidental `runState` copies. That addresses the real underlying worry.

### Critical #2 — `Sink.Send` blocking semantics

**He's half-right, for the wrong reason.** His specific example (`QuestionEvent` getting dropped) doesn't hold: `QuestionEvent` is informational-only per M2; the actual reply path is `HITLHandler.AskUser`, which blocks. A dropped `QuestionEvent` means an observer misses a log line, not that a user's question vanishes.

That said, "Send must not block indefinitely, and BufferedSink drops on full" *is* a confusing pair for callers who need reliable delivery. Tighten the contract, don't invert it:

- `Sink.Send` is non-blocking by contract.
- `BufferedSink` drops on full (documented, with a metric surfaced via a `DroppedCount()` method).
- Callers who need guaranteed delivery supply their own `Sink` (e.g., bounded channel with blocking send + caller-owned timeout). The harness doesn't mandate a delivery policy.

**Do not** introduce `Sink.WithTimeout`. That's policy in the interface.

### Important #7 — `TokenAccountant` as single source of truth

**He's wrong.** Folding budget enforcement into middleware creates ordering dependencies (what if a caller replaces the accountant? the budget check disappears silently) and couples a correctness-critical check to an observability concern. The current split is exactly right: the loop enforces (non-negotiable), the accountant observes (swappable).

Keep the invariant test as a `sleipnirtest` helper. He's right it can't run in prod, but that's not the point — the invariant proves the two counters don't drift *in CI*, which is enough.

**Response:** "Budget enforcement is a harness correctness concern, not a middleware concern. Middleware is swappable; the budget check is not. Keeping them separate is deliberate."

### Important #8 — `ContextRewriter` "can't modify the request"

**He misread the signature.** `RewriteBeforeLLMCall(ctx context.Context, req *LLMRequest) error` — `req` is a pointer. Mutation is the intended mechanism; the compactor already relies on it. No change needed.

Point this out politely. Don't let it slide — if he thinks the compactor works by side channel, his mental model of the design is off, and that will bite future reviews.

### Minor #9 — `SystemPromptString` parallel field

Reject. Function-call overhead is noise against an LLM round-trip. Adding a parallel field doubles the API surface for a non-problem. `func(AgentInput) string` that ignores its argument is a two-line closure.

### Idiomatic Go — `sleipnir/test` vs `sleipnirtest`

Reject. `xxxtest` is the standard Go pattern (`httptest`, `iotest`, `fstest`, `synctest`). Director's the one off-idiom here.

---

## Where the director is right (accept)

### Critical #3 — freeze latch too strict for SaaS

Accept, with a twist. His "rebuild tool index every Run" is fine but overengineered for v1. Simpler:

- Keep `RegisterAgent` → `ErrHarnessFrozen` as the **default** post-first-Run behavior.
- Add `Config.AllowLateRegistration bool` (default `false`). When true, `RegisterAgent` is safe concurrent with `Run`, backed by a `sync.Map` of specs with a per-`Run` snapshot taken at loop entry.
- Hot-reload without this flag: construct a new `Harness`, atomic-swap in the caller. Document in the README.

Default-safe, opt-in flexible.

### Critical #4 — `ModelRouter.Resolve` per iteration

Accept. Two changes:

1. Document in the `ModelRouter` godoc that implementations SHOULD cache unless they have a reason not to.
2. Ship a `CachedRouter` wrapper in the main package (caches per `(Run, agentName)` for the duration of a `Run`). Callers who want per-iteration resolution don't wrap.

Do **not** force per-Run caching at the harness level — canary-rollout routers legitimately want per-call resolution.

### Important #5 — `ExtraTools` inheritance is a security footgun

Accept the concern, adjust the fix. His `ExtraToolsInheritance map[string][]Tool` is overengineered. Simpler:

- Add `RunInput.InheritExtraTools bool` (default `true` — matches the ergonomic default we agreed on previously).
- Document the security consideration: inheriting `AskUserTool` means a prompt-injected sub-agent can ask the user anything. Mitigation lives in the `HITLHandler` implementation (validate/scope/rate-limit). Callers who need per-sub-agent scoping set `InheritExtraTools: false` and pass tools explicitly via their `AgentAsTool` wrapper or sub-agent's own `spec.Tools`.

Default-ergonomic, opt-in secure. His `default false` would force every Urd sub-agent to opt into HITL — that's noise.

### Important #6 — todo race across parallel same-name sub-agents

Accept. Real bug. If a parent dispatches two concurrent sub-agent tool_use blocks targeting the same sub-agent name, they share a todo list keyed by agent name and race.

Fix: key todos by **`ToolCallID`** (already unique per dispatch, already flows through the event stream) instead of agent name. Top-level run uses a synthetic root ID. Update §5.2, §10, and `TodoEvent` to carry the ID.

### Minor #10 — `MaxTotalTokens` scope

Accept as a doc fix, not an API change. §11 already describes this; make the example explicit: "100k budget, 5 sub-agents @ 30k each → run halts when *cumulative* usage hits 100k, regardless of which agent consumed it." No new `AgentSpec.MaxTokens` field — scope creep.

### Idiomatic Go — `BufferedSink` return type

Accept. `func BufferedSink(...) *BufferedSink` with `(*BufferedSink).Events() <-chan Event` is cleaner and lets us add methods (`DroppedCount()` from Critical #2 above).

### Idiomatic Go — verify `anyllm.Message` exists

Accept. Trivial to verify before the first milestone PR. Not a design-stage concern.

---

## Defer (director's call is fine, but v2 scope)

- **#11 `ToolResult.StructuredData`** — he agrees v2. No action.
- **#12 `mcpadapter.Watch`** — v2. §16 already lists live subscription as a non-goal; keep it.
- **Idiomatic — `NewTypedTool` godoc** — add a one-line godoc explaining why it returns `error` (reflection can fail on cycles/unsupported types). Cheap.

---

## Updates for design v3

1. §5.2: keep `atomic.Int64`; add a note explaining the multi-goroutine write model. Add CI note for `go vet copylocks`.
2. §6: tighten `Sink.Send` contract to non-blocking, non-policy. Document `BufferedSink` drop counter.
3. §4 / §12: add `Config.AllowLateRegistration`; document hot-reload-via-swap.
4. §4: add `CachedRouter` to shipped helpers; update `ModelRouter` godoc with caching guidance.
5. §5.1 / §4: add `RunInput.InheritExtraTools bool` (default `true`); update inheritance table with security note.
6. §5.2 / §10 / §6: key todos by `ToolCallID`; add the ID to `TodoEvent`.
7. §11: add the cumulative-budget example.
8. §4: `BufferedSink` returns `*BufferedSink` with exported methods.
9. §8: one-line godoc on `NewTypedTool` explaining `error` return.
10. Appendix A: add v2→v3 changelog rows for each.

**Items to explicitly not change, with rationale in the doc** so future reviewers don't re-litigate:

- `atomic.Int64` on `tokensUsed` (concurrent writers by design).
- `TokenAccountant` as observer, not source of truth (enforcement ≠ observability).
- `ContextRewriter` mutation via pointer (works as-is; director misread).
- `SystemPrompt` func-only (no parallel string field).
- `sleipnirtest` package name (matches Go stdlib convention).

---

## Questions for you before v3

1. **`InheritExtraTools` default true?** Confirm or argue. I want reasoning, not capitulation.
2. **Todo keying by `ToolCallID`:** any objection? Alternative is a fresh `invocationID`, but `ToolCallID` already exists.
3. **`CachedRouter` placement:** main package or a subpackage? I lean main — it's a wrapper, not middleware.
4. **`AllowLateRegistration`:** does Urd actually need this for v1? If not, defer the flag. Don't build it speculatively.

---

## One meta-point

The director's review is about 60% correct. That's normal — directors review at volume and miss details. Our job is to read carefully, accept what's right, push back on what's wrong, and have evidence ready. Seniority is not authority on technical correctness. A well-argued pushback earns more respect than silent acceptance of a wrong call.

Specifically: if you quietly swap `atomic.Int64` for a mutex because he said so, you'll introduce a real bug (the mutex either gets held across the LLM call, or acquired/released around a counter bump — equivalent to atomic but slower) to fix an imaginary one. That's worse than the current design.

Update v2 → v3 with the changes above, flag the explicit non-changes in the changelog with rationale, and re-submit.
