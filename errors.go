package sleipnir

import "errors"

// Harness lifecycle errors.
var (
	ErrHarnessFrozen      = errors.New("sleipnir: harness is frozen; RegisterAgent called after first Run with AllowLateRegistration disabled")
	ErrAgentNotRegistered = errors.New("sleipnir: agent not registered")
	ErrToolNameCollision  = errors.New("sleipnir: duplicate tool name detected")
)

// Run budget errors.
var (
	ErrIterationBudget = errors.New("sleipnir: agent loop exhausted MaxIterations without a final text response")
	ErrTokenBudget     = errors.New("sleipnir: cumulative token usage exceeded RunInput.MaxTotalTokens")
)

// Human-in-the-loop errors.
var (
	ErrHITLTimeout   = errors.New("sleipnir: HITL handler did not return within SLEIPNIR_HITL_TIMEOUT")
	ErrHITLCancelled = errors.New("sleipnir: HITL handler returned because context was cancelled")
)

// Middleware errors.
var (
	ErrCompactionFailed = errors.New("sleipnir: compactor middleware failed; run proceeds uncompacted")
)
