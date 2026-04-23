package sleipnirtest

import (
	"testing"

	"sleipnir.dev/sleipnir"
)

var _ sleipnir.Sink = (*EventCollector)(nil)

func TestCollectorSend(t *testing.T) {
	tests := []struct {
		name   string
		events []sleipnir.Event
	}{
		{
			name:   "single event",
			events: []sleipnir.Event{&sleipnir.TokenEvent{}},
		},
		{
			name:   "multiple events",
			events: []sleipnir.Event{&sleipnir.TokenEvent{}, &sleipnir.TokenEvent{}, &sleipnir.TokenEvent{}},
		},
		{
			name:   "empty",
			events: []sleipnir.Event{},
		},
		{
			name:   "nil event",
			events: []sleipnir.Event{nil},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			collector := NewEventCollector()

			// Send events
			for _, event := range tt.events {
				collector.Send(event)
			}

			// Get events
			got := collector.Events()

			// Verify count
			if len(got) != len(tt.events) {
				t.Errorf("Expected %d events, got %d", len(tt.events), len(got))
			}

			// Verify each event (using reflection or direct comparison)
			for i, expected := range tt.events {
				if i >= len(got) {
					break
				}
				if expected != got[i] {
					t.Errorf("Event %d: expected %v, got %v", i, expected, got[i])
				}
			}
		})
	}
}

func TestCollectorByAgent(t *testing.T) {
	collector := NewEventCollector()

	// Create events for different agents
	agentEvents := []sleipnir.Event{
		sleipnir.AgentStartEvent{AgentName: "alice"},
		sleipnir.TokenEvent{AgentName: "alice"},
		sleipnir.AgentEndEvent{
			AgentName: "alice",
		},
		sleipnir.AgentStartEvent{AgentName: "bob"},
		sleipnir.TokenEvent{AgentName: "bob"},
		sleipnir.ThinkingEvent{AgentName: "bob"},
		sleipnir.AgentEndEvent{
			AgentName: "bob",
		},
	}

	// Send all events
	for _, event := range agentEvents {
		collector.Send(event)
	}

	// Filter by agent "alice" -- should return exactly 3 events (AgentStart, Token, AgentEnd)
	aliceEvents := collector.ByAgent("alice")
	if len(aliceEvents) != 3 {
		t.Errorf("Expected 3 events for agent 'alice', got %d", len(aliceEvents))
	}

	// Verify that only `alice` events were returned
	for _, event := range aliceEvents {
		namer, ok := event.(sleipnir.AgentNamer)
		if !ok {
			t.Errorf("Event does not implement AgentNamer: %T", event)
			continue
		}
		if namer.EventAgent() != "alice" {
			t.Errorf("Expected agent 'alice', got '%s'", namer.EventAgent())
		}
	}

	// Filter by agent "bob" -- should return exactly 4 events (AgentStart, Token, Thinking, AgentEnd)
	bobEvents := collector.ByAgent("bob")
	if len(bobEvents) != 4 {
		t.Errorf("Expected 4 events for agent 'bob', got %d", len(bobEvents))
	}

	// Verify that only `bob` events were returned
	for _, event := range bobEvents {
		namer, ok := event.(sleipnir.AgentNamer)
		if !ok {
			t.Errorf("Event does not implement AgentNamer: %T", event)
			continue
		}
		if namer.EventAgent() != "bob" {
			t.Errorf("Expected agent 'bob', got '%s'", namer.EventAgent())
		}
	}

	// Filter by non-existent agent -- should return no events
	charlieEvents := collector.ByAgent("charlie")
	if len(charlieEvents) != 0 {
		t.Errorf("Expected 0 events for agent 'charlie', got %d", len(charlieEvents))
	}
}

func TestByType(t *testing.T) {
	collector := NewEventCollector()

	// Send some events: 3x TokenEvent, 1x (AgentStart, AgentEnd, Thinking)
	eventsToSend := []sleipnir.Event{
		sleipnir.AgentStartEvent{},
		sleipnir.AgentEndEvent{},
		sleipnir.TokenEvent{},
		sleipnir.ThinkingEvent{},
		sleipnir.TokenEvent{},
		sleipnir.TokenEvent{},
	}
	for _, event := range eventsToSend {
		collector.Send(event)
	}

	// Filter by type TokenEvent -- should return exactly 3 events
	tokenEvents := ByType[sleipnir.TokenEvent](collector)
	if len(tokenEvents) != 3 {
		t.Errorf("Expected 3 TokenEvents, got %d", len(tokenEvents))
	}

}

func TestCollectorCheckCompleted(t *testing.T) {
	tests := []struct {
		name      string
		events    []sleipnir.Event
		shouldErr bool
	}{
		{
			name: "returns nil when StopDone present",
			events: []sleipnir.Event{
				sleipnir.AgentEndEvent{Stopped: sleipnir.StopDone},
			},
			shouldErr: false,
		},
		{
			name: "returns error when no StopDone",
			events: []sleipnir.Event{
				sleipnir.AgentStartEvent{},
			},
			shouldErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			collector := NewEventCollector()
			for _, event := range tt.events {
				collector.Send(event)
			}

			err := collector.CheckCompleted()
			if (err != nil) != tt.shouldErr {
				t.Errorf("Expected error: %v, got: %v", tt.shouldErr, err)
			}
		})
	}
}
