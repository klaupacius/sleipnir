package sleipnir

import (
	"context"
	"sync/atomic"
)

type Sink interface{ Send(Event) }

type BufferedSink struct {
	ch      chan Event
	dropped atomic.Int64
}

func NewBufferedSink(ctx context.Context, size int) *BufferedSink {
	s := &BufferedSink{ch: make(chan Event, size)}
	go func() {
		<-ctx.Done()
		close(s.ch)
	}()
	return s
}

func (s *BufferedSink) Send(e Event) {
	select {
	case s.ch <- e:
	default:
		s.dropped.Add(1)
	}
}

func (s *BufferedSink) Events() <-chan Event { return s.ch }
func (s *BufferedSink) DroppedCount() int64  { return s.dropped.Load() }
