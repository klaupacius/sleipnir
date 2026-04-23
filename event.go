package sleipnir

import "encoding/json"

type Event interface{ eventMarker() }

type TodoStatus string

const (
	TodoPending    TodoStatus = "pending"
	TodoInProgress TodoStatus = "in_progress"
	TodoDone       TodoStatus = "done"
)

type TodoItem struct {
	ID     string
	Text   string
	Status TodoStatus
}

type AgentStartEvent struct{ AgentName, ParentName string }

func (e AgentStartEvent) eventMarker() {}

type AgentEndEvent struct {
	AgentName string
	Usage     Usage
	Stopped   StopReason
}

func (e AgentEndEvent) eventMarker() {}

type TokenEvent struct{ AgentName, Text string }

func (e TokenEvent) eventMarker() {}

type ThinkingEvent struct{ AgentName, Text string }

func (e ThinkingEvent) eventMarker() {}

type ToolCallEvent struct {
	AgentName, ToolCallID, ToolName string
	Args                            json.RawMessage
}

func (e ToolCallEvent) eventMarker() {}

type ToolResultEvent struct {
	AgentName, ToolCallID string
	Result                string
	IsError               bool
}

func (e ToolResultEvent) eventMarker() {}

type QuestionEvent struct{ AgentName, QuestionID, Question string }

func (e QuestionEvent) eventMarker() {}

type TodoEvent struct {
	AgentName, ToolCallID string
	Todos                 []TodoItem
}

func (e TodoEvent) eventMarker() {}

type ErrorEvent struct {
	AgentName string
	Err       error
}

func (e ErrorEvent) eventMarker() {}

// Exported so sleipnirtest can use it:
type AgentNamer interface {
	EventAgent() string
}

func (e AgentStartEvent) EventAgent() string { return e.AgentName }
func (e AgentEndEvent) EventAgent() string   { return e.AgentName }
func (e TokenEvent) EventAgent() string      { return e.AgentName }
func (e ThinkingEvent) EventAgent() string   { return e.AgentName }
func (e ToolCallEvent) EventAgent() string   { return e.AgentName }
func (e ToolResultEvent) EventAgent() string { return e.AgentName }
func (e QuestionEvent) EventAgent() string   { return e.AgentName }
func (e TodoEvent) EventAgent() string       { return e.AgentName }
func (e ErrorEvent) EventAgent() string      { return e.AgentName }
