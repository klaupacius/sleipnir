package sleipnirtest

import (
	"testing"

	"sleipnir.dev/sleipnir"
)

func TestTodoCollectorReceives(t *testing.T) {
	c := NewTodoCollector()
	todos := []sleipnir.TodoItem{
		{ID: "1", Text: "do something", Status: sleipnir.TodoPending},
	}
	c.Send(sleipnir.TodoEvent{AgentName: "agent", ToolCallID: "call-1", Todos: todos})

	got := c.ByToolCallID("call-1")
	if len(got) != 1 {
		t.Fatalf("expected 1 item, got %d", len(got))
	}
	if got[0].ID != "1" || got[0].Text != "do something" {
		t.Errorf("unexpected todo item: %+v", got[0])
	}
}

func TestTodoCollectorLastWins(t *testing.T) {
	c := NewTodoCollector()
	first := []sleipnir.TodoItem{{ID: "1", Text: "first", Status: sleipnir.TodoPending}}
	second := []sleipnir.TodoItem{
		{ID: "1", Text: "first", Status: sleipnir.TodoDone},
		{ID: "2", Text: "second", Status: sleipnir.TodoPending},
	}
	c.Send(sleipnir.TodoEvent{AgentName: "agent", ToolCallID: "call-1", Todos: first})
	c.Send(sleipnir.TodoEvent{AgentName: "agent", ToolCallID: "call-1", Todos: second})

	got := c.ByToolCallID("call-1")
	if len(got) != 2 {
		t.Fatalf("expected 2 items after last-wins update, got %d", len(got))
	}
	if got[0].Status != sleipnir.TodoDone {
		t.Errorf("expected first item status=done, got %s", got[0].Status)
	}
}

func TestTodoCollectorIgnoresOtherEvents(t *testing.T) {
	c := NewTodoCollector()
	c.Send(sleipnir.TokenEvent{AgentName: "agent", Text: "hello"})
	c.Send(sleipnir.AgentStartEvent{AgentName: "agent"})

	all := c.All()
	if len(all) != 0 {
		t.Errorf("expected no entries, got %d", len(all))
	}
}
