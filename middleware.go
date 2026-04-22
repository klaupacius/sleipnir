package sleipnir

// Middleware is a marker interface for middleware values. A middleware may
// implement any subset of ContextRewriter, LLMObserver, ToolObserver, and
// RetryPolicy. The harness discovers capabilities via type assertion.
type Middleware interface{ middleware() }
