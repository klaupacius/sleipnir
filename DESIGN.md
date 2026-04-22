# *Sleipnir*: A Minimal Go Multi-Agent Harness

In Norse mythology, *Sleipnir* is Odin's eight-legged horse, on which he rides across sky and sea.

## Concept

A focused, bare-bones Go harness for multi-agent LLM orchestration, designed first for the Urd SaaS research assistant and generalisable to similar applications. An orchestrator agent processes the user's query and delegates research tasks to one or more sub-agents that run in parallel. Context management and compression, todo/task list management, and basic logging and observability are included.

## Tech Stack

- Go 1.25+
- `any-llm-go` (https://github.com/mozilla-ai/any-llm-go) for multi-provider support (other libraries may be better?)


## Design Principles

1. Follow the "Twelve-Factor App" design principles as far as practicable. E.g. all config is stored as environment variables.
2. A thin orchestration layer built on explicit interfaces. The agentic loop is simple Go code, not hidden behind framework "magic".
3. Parallel execution independent of LLM choice.
4. Context management, observability, and token accounting implemented as pluggable "middleware".
5. Sub-agents are tools: From the parent's perspective, a sub-agent is called via normal `tool_use`. No special dispatch or recursive sub-agents.
6. The harness owns no data: Message history, document retrieval, and embeddings all belong to the application layer.