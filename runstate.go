package sleipnir

import (
	"sync"
	"sync/atomic"
)

// runState holds per-Run shared state threaded through the agent loop and all
// recursive sub-agent calls. Always passed as *runState; must never be copied.
type runState struct {
	// Concurrent writes from parent loop + parallel sub-agent goroutines.
	tokensUsed  atomic.Int64
	tokenBudget int64 // 0 = unlimited

	// Keyed by ToolCallID. Populated by TodoWriteTool / TodoReadTool (chunk 16).
	todos sync.Map // map[string][]TodoItem
}

func newRunState(maxTotalTokens int) *runState {
	return &runState{tokenBudget: int64(maxTotalTokens)}
}

func (s *runState) addTokens(u Usage) {
	s.tokensUsed.Add(u.TotalTokens)
}

// overBudget reports whether cumulative token usage has reached the budget.
// Returns false when tokenBudget is 0 (unlimited).
func (s *runState) overBudget() bool {
	return s.tokenBudget > 0 && s.tokensUsed.Load() >= s.tokenBudget
}
