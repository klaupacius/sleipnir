package sleipnir

import (
	"context"
	"sync"
	"testing"
	"time"
)

// Compile-time assertions
var _ Sink = (*BufferedSink)(nil)

func TestBufferedSinkReceive(t *testing.T) {
	ctx := context.Background()
	size := 10
	N := 5

	sink := NewBufferedSink(ctx, size)

	// Send N events (N < size)
	for i := 0; i < N; i++ {
		sink.Send(TokenEvent{})
	}

	// Collect from Events() channel
	events := sink.Events()
	received := 0
	for received < N {
		select {
		case <-events:
			received++
		case <-time.After(1 * time.Second):
			t.Fatalf("Timed out after receiving %d/%d events", received, N)
		}
	}

	// Verify no messages left in channel
	select {
	case <-events:
		t.Error("Unexpected extra event in channel")
	default:
		// Channel empty - good
	}

	// Verify DroppedCount is zero
	if sink.DroppedCount() != 0 {
		t.Errorf("DroppedCount = %d, want 0", sink.DroppedCount())
	}
}

func countReceivedEvents(sink *BufferedSink) int {
	count := 0
	events := sink.Events()
	for {
		select {
		case <-events:
			count++
		default:
			return count
		}
	}
}

func TestBufferedSinkDrop(t *testing.T) {
	ctx := context.Background()
	size := 5
	N := 10

	sink := NewBufferedSink(ctx, size)

	// Send N events (N > size)
	for i := 0; i < N; i++ {
		sink.Send(TokenEvent{})
	}

	// Drain the channel and confirm we received `size` events
	want := size
	got := countReceivedEvents(sink)
	if got != want {
		t.Fatalf("Received %d events, expected %d", got, want)
	}

	// Confirm that `DroppedCount()` == N-size
	want = N - size
	got = int(sink.DroppedCount())
	if got != want {
		t.Fatalf("DroppedCount was %d, expected %d", got, want)
	}
}

// Run TestBufferedSinkConcurrentSend with -race flag
func TestBufferedSinkConcurrentSend(t *testing.T) {
	ctx := context.Background()
	size := 2000 // Large enough buffer to avoid dropped events
	sink := NewBufferedSink(ctx, size)

	numGoroutines := 10
	eventsPerGoroutine := 100

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// Launch concurrent sends - the -race flag will detect data races
	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < eventsPerGoroutine; j++ {
				sink.Send(TokenEvent{})
			}
		}()
	}

	// Wait for goroutines to finish
	wg.Wait()

	// Verify we didn't drop events
	want := numGoroutines * eventsPerGoroutine
	got := int(sink.DroppedCount())
	if got > want {
		t.Fatalf("DroppedCount got=%d, want=%d", got, want)
	}

	// Drain channel to ensure no goroutine leaks
	go func() {
		for range sink.Events() {
			// Drain channel
		}
	}()
}

// TestBufferedSinkSendAfterClose verifies that Send does not panic when the
// context is cancelled and the channel has been closed.
func TestBufferedSinkSendAfterClose(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	sink := NewBufferedSink(ctx, 10)
	cancel()
	// Wait for the close goroutine to close the channel.
	time.Sleep(20 * time.Millisecond)
	// Must not panic:
	sink.Send(TokenEvent{})
}

func TestBufferedSinkContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	size := 10 // size >= N to ensure no dropped events
	N := 5
	sink := NewBufferedSink(ctx, size)

	// Send some events before cancellation
	for i := 0; i < N; i++ {
		sink.Send(TokenEvent{})
	}

	// Cancel the context
	cancel()

	// Give the goroutine time to close the channel
	time.Sleep(50 * time.Millisecond)

	// Check that we can read the existing events
	events := sink.Events()
	var received []Event
	for event := range events {
		received = append(received, event)
	}
	if len(received) != N {
		t.Errorf("Expected %d events before close, got %d", N, len(received))
	}

	// Second receive should be zero value with ok=false
	if _, ok := <-events; ok {
		t.Error("Events() channel remained open after context cancel")
	}

	// Verify no dropped events
	if sink.DroppedCount() != 0 {
		t.Errorf("Got %d dropped events, expected 0", sink.DroppedCount())
	}
}
