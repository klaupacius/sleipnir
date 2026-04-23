# Sleipnir — Claude Code context

## What this project is

A Go multi-agent LLM orchestration harness. Replaces Pydantic AI in the Urd SaaS stack. Apache-2.0, open-source. Module path: `sleipnir.dev/sleipnir`. Go 1.25+.

## Key documents

| Document | Purpose |
|---|---|
| `docs/plans/sleipnir-architecture-design-v3.md` | Approved architecture — source of truth for public API and design decisions |
| `/home/jostein/.claude/plans/the-architecture-design-docs-plans-sleip-harmonic-catmull.md` | Chunk-by-chunk implementation plan — **read this before starting any planning or implementation work** |

## Implementation status

Chunks 1–3 are complete. Chunks 4–8 are being implemented. **Next planning session resumes at Chunk 9.**

The full chunk list is in the implementation plan file above.

## Planning process

This project follows a deliberate chunk-by-chunk planning process with a junior developer doing the implementation:

1. Detail one chunk at a time (goal, files, types/logic, tests, acceptance criteria)
2. Present to the user and ask for comments
3. Refine based on feedback, then move to the next chunk
4. **Do not detail multiple chunks at once without user approval between them**

## Locked design decisions

See `/home/jostein/.claude/projects/-home-jostein-Projects-sleipnir/memory/planning_decisions.md` for a full log. Key ones:

- `OmitExtraToolsInheritance bool` (not `InheritExtraTools`) — zero = inherit
- `logLevelSet bool` unexported field on `Config` to disambiguate `slog.LevelInfo == 0`
- `AgentNamer` interface with `EventAgent() string` method for `ByAgent` filtering
- `StopIterationBudget`: return both `RunOutput` and `ErrIterationBudget`
- `anyllm.Provider.Completion(ctx, CompletionParams) (*ChatCompletion, error)` confirmed
- Tool infra errors are isolated — do not abort the run

## Dependencies confirmed

| Package | Role |
|---|---|
| `github.com/mozilla-ai/any-llm-go` | LLM provider boundary (added in chunk 3) |
| `github.com/invopop/jsonschema` | Schema reflection for `NewTypedTool` (added in chunk 7) |
| `github.com/modelcontextprotocol/go-sdk` | MCP client — `mcpadapter/` only (chunk 19) |

## Running tests

```sh
go test -race ./...
go vet ./...
```
