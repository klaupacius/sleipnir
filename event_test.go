package sleipnir

// Compile-time assertions for Event types
var (
	_ Event = AgentStartEvent{}
	_ Event = AgentEndEvent{}
	_ Event = TokenEvent{}
	_ Event = ThinkingEvent{}
	_ Event = ToolCallEvent{}
	_ Event = ToolResultEvent{}
	_ Event = QuestionEvent{}
	_ Event = TodoEvent{}
	_ Event = ErrorEvent{}
)

var _ AgentNamer = AgentStartEvent{}
