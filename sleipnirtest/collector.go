package sleipnirtest

import (
	"errors"
	"sync"

	"sleipnir.dev/sleipnir"
)

// EventCollector implements Sink and accumulates all sent events.
// Safe for concurrent use.
type EventCollector struct {
	mu     sync.Mutex
	events []sleipnir.Event
}

func NewEventCollector() *EventCollector { return new(EventCollector) }

func (c *EventCollector) Send(e sleipnir.Event) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.events = append(c.events, e)
}

// ByAgent returns all events where the AgentName field equals name.
// Uses reflection so it works across all event types.
func (c *EventCollector) ByAgent(name string) []sleipnir.Event {
	c.mu.Lock()
	defer c.mu.Unlock()
	var result []sleipnir.Event
	for _, e := range c.events {
		if n, ok := e.(sleipnir.AgentNamer); ok && n.EventAgent() == name {
			result = append(result, e)
		}
	}
	return result
}

// Events returns a snapshot of all collected events.
func (c *EventCollector) Events() []sleipnir.Event {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.events
}

// CheckCompleted returns an error if no AgentEndEvent with StopDone was received.
func (c *EventCollector) CheckCompleted() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, e := range c.events {
		if endEvent, ok := e.(sleipnir.AgentEndEvent); ok {
			if endEvent.Stopped == sleipnir.StopDone {
				return nil
			}
		}
	}
	return errors.New("no AgentEndEvent with StopDone was received")
}

// ByType returns all events in c that are of type T.
func ByType[T sleipnir.Event](c *EventCollector) []T {
	c.mu.Lock()
	defer c.mu.Unlock()

	var result []T
	for _, e := range c.events {
		if typed, ok := e.(T); ok {
			result = append(result, typed)
		}
	}
	return result
}
