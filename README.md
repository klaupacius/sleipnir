# Sleipnir

A minimal Go multi-agent LLM orchestration harness.

> In Norse mythology, *Sleipnir* is Odin's eight-legged horse, on which he can
> ride across sky and sea.

## Status

**Pre-alpha.** Design complete; implementation in progress. Not yet usable.

See [`docs/plans/2026-04-22-sleipnir-architecture-design.md`](docs/plans/2026-04-22-sleipnir-architecture-design.md)
for the architecture and [`DESIGN.md`](DESIGN.md) for the concept summary.

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

## License

[Apache-2.0](LICENSE).
