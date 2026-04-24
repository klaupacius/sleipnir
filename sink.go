package sleipnir

import (
	"context"
	"sync"
	"sync/atomic"
)

type Sink interface{ Send(Event) }

type BufferedSink struct {
	mu      sync.Mutex
	ch      chan Event
	closed  bool
	dropped atomic.Int64
}

func NewBufferedSink(ctx context.Context, size int) *BufferedSink {
	s := &BufferedSink{ch: make(chan Event, size)}
	go func() {
		<-ctx.Done()
		s.mu.Lock()
		s.closed = true
		close(s.ch)
		s.mu.Unlock()
	}()
	return s
}

func (s *BufferedSink) Send(e Event) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		s.dropped.Add(1)
		return
	}
	select {
	case s.ch <- e:
	default:
		s.dropped.Add(1)
	}
}

func (s *BufferedSink) Events() <-chan Event { return s.ch }
func (s *BufferedSink) DroppedCount() int64  { return s.dropped.Load() }
