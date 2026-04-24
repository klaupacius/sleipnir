package sleipnirtest

import (
	"maps"
	"sync"

	"sleipnir.dev/sleipnir"
)

// TodoCollector implements Sink and accumulates TodoEvents.
// After a run, ByToolCallID returns the final todo list for each invocation.
type TodoCollector struct {
	mu     sync.Mutex
	latest map[string][]sleipnir.TodoItem
}

func NewTodoCollector() *TodoCollector { return new(TodoCollector) }

func (c *TodoCollector) Send(e sleipnir.Event) {
	if te, ok := e.(sleipnir.TodoEvent); ok {
		c.mu.Lock()
		defer c.mu.Unlock()
		if c.latest == nil {
			c.latest = make(map[string][]sleipnir.TodoItem)
		}
		c.latest[te.ToolCallID] = te.Todos
	}
}

func (c *TodoCollector) ByToolCallID(id string) []sleipnir.TodoItem {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.latest[id]
}

func (c *TodoCollector) All() map[string][]sleipnir.TodoItem {
	c.mu.Lock()
	defer c.mu.Unlock()
	result := make(map[string][]sleipnir.TodoItem, len(c.latest))
	maps.Copy(result, c.latest)
	return result
}

var _ sleipnir.Sink = (*TodoCollector)(nil)
