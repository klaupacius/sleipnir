## Review of Sleipnir Architecture Design (v2)

This is a **remarkably solid design** for a junior engineer's first solo project. The mentor's influence shows—most of the structural concerns I'd normally raise are already addressed in the appendix. That said, I have several practical concerns from real-world Go library maintenance.

---

### Critical Issues (Must Fix Before Implementation)

**1. `runState.tokensUsed` as `atomic.Int64` is the wrong abstraction**

The design says "sole writer is the agent loop" but then uses an atomic. Those are contradictory. If there's one writer, use plain `int64` with the loop holding the write mutex (which you already need for other runState mutations). Atomics signal "multiple writers" to future maintainers. Worse, you're copying `runState` (via argument passing) but atomics shouldn't be copied. This will cause obscure bugs.

**Fix:** Make `runState` a struct with a plain `mu sync.Mutex` and `tokensUsed int64`. Keep the one-writer rule but protect with mutex for safety when invariants inevitably break.

**2. `Sink.Send` non-blocking requirement is impossible to satisfy safely**

You require `Send` not to block indefinitely, but also require `BufferedSink` to drop on full buffer. This means callers *cannot* rely on events being delivered. In production, you *will* drop `QuestionEvent`s before the handler runs, leading to confused users wondering why their question wasn't logged.

**Fix:** Make `Send` blocking with a per-Sink timeout. Add `Sink.WithTimeout(time.Duration)` if needed. Drop the "must not block indefinitely" requirement—replace with "should not block longer than reasonable for the deployment."

**3. `RegisterAgent` after first `Run` = error is too strict**

You'll regret this within a week of Urd deployment. Configuration reloading is table stakes for SaaS. The freeze latch forces a restart on every agent definition change.

**Fix:** Allow registration after `Run` but require explicit `harness.Freeze()` for callers who want the safety. Or make registration safe concurrent with `Run` (store in `sync.Map`, rebuild tool index on each `Run` with acceptable overhead).

**4. `ModelRouter.Resolve` called every iteration is wasteful**

Real routers will hit databases, feature flags, or external services. Calling per iteration (potentially 20x per agent, times sub-agents) is a DoS risk.

**Fix:** Document that implementations MUST cache unless they have a reason not to. Add `CachedRouter` wrapper. Or change the contract to allow the harness to cache for the duration of a `Run`.

---

### Important Concerns (Fix During Implementation)

**5. Sub-agent inheritance of `ExtraTools` is a security footgun**

If a parent adds `AskUserTool` to `ExtraTools`, all sub-agents inherit it. That means a sub-agent invoked with a malicious prompt could ask the user for credentials. The parent agent has no visibility into which sub-agents get which tools.

**Fix:** Make `ExtraTools` inheritance explicit: `ExtraToolsInheritance map[string][]Tool` (agent name → tools) or a simpler `InheritTools bool` field. Default to `false` for security.

**6. `TodoWriteTool` full-replace semantics will cause race conditions**

LLM emits `todo_write` with new list. Harness replaces. Another LLM call in same iteration? No, you process sequentially. But parallel sub-agents writing to same agent's todo list? That's a real race—two sub-agents could both read, modify, and write simultaneously.

**Fix:** Document that todo state is not safe for concurrent writes from parallel sub-agents (or add a mutex keyed by agent name in `runState`).

**7. `TokenAccountant` vs `runState.tokensUsed` invariant is untestable in production**

You can't assert this in production without performance impact. The invariant is good for tests, but you're relying on it as a correctness guarantee.

**Fix:** Make `TokenAccountant` the *only* source of truth. Remove `runState.tokensUsed` entirely. The loop calls `TokenAccountant.AddUsage()` after each LLM call. Budget check reads from accountant. One writer, one reader, no invariant to assert.

**8. `ContextRewriter.RewriteBeforeLLMCall` can't modify the request meaningfully**

It receives `*LLMRequest` but can only return an error. What if it needs to replace the entire message history (e.g., compaction)?

**Fix:** Allow mutation of `LLMRequest.Messages` (or provide a `ReplaceMessages` method). The current design forces compactor to work by side channel.

---

### Minor Issues (Nice to Fix)

**9. `AgentSpec.SystemPrompt` as a function is inflexible**

Every agent invocation requires re-executing the function. For static prompts, this is wasteful. For prompts that depend on `AgentInput`, fine.

**Suggestion:** Add `SystemPromptString string` field, and only call the function if `SystemPrompt` is non-nil.

**10. `MaxTotalTokens` at `RunInput` level is surprising**

If a parent has 100k budget and spawns 5 sub-agents, each consuming 30k, the parent's loop will stop at 100k total. That's fine, but not what users expect. They expect per-agent budgets.

**Fix:** Add `AgentSpec.MaxTokens` (per-invocation) and `RunInput.MaxTotalTokens` (parent+subs combined). Or document clearly and provide examples.

**11. No way to get sub-agent results except through final tool_result**

The parent sees only the final text. That's by design (isolation), but you'll need structured results eventually.

**Defer:** Add `ToolResult.StructuredData any` in v2.

**12. `mcpadapter` snapshot semantics are wrong for long-lived servers**

MCP servers can add/remove tools dynamically. Your doc says "consumers must re-call LoadTools" but there's no way to know when to do that.

**Fix:** Add `mcpadapter.Watch(ctx, client, callback)` that polls or uses MCP notifications (when available).

---

### Idiomatic Go Concerns

- `NewTypedTool[T]` returning `(Tool, error)` is fine, but document clearly why it's not `MustNewTypedTool` (schema reflection failures).
- `BufferedSink` should return `*BufferedSink` not `(Sink, <-chan Event)`—the channel is part of the implementation.
- `sleipnirtest` package name is fine but `sleipnir/test` would be more Go-like (`testing` package doesn't own the namespace).
- `anyllm.Message` — verify this type exists in the actual dependency before committing.

---

### Overall Assessment

**Ready for implementation with the critical fixes above.** 

This is a v1 library that will ship. The junior engineer has done exceptional work incorporating feedback. The design's strengths (typed events, clean tool interface, explicit budgets, no hidden state) outweigh the issues.

**Required before implementation starts:**
1. Fix `runState` mutex/atomic confusion
2. Clarify `Sink` blocking semantics
3. Remove or make optional the freeze-latch

**Expected in first PR after basic loop works:**
4. Security review of tool inheritance
5. `TokenAccountant` as single source of truth

Tell the junior engineer she's ready to start coding, but schedule a 30-minute review after she implements the basic loop to catch any emergent issues. And buy her mentor a coffee—this is good work.