package retry

import (
	"context"
	"errors"
	"testing"
	"time"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

func TestDefaultRetryPolicyRetryableErrors(t *testing.T) {
	p := DefaultRetryPolicy{}
	ctx := context.Background()

	for _, err := range []error{anyllm.ErrRateLimit, anyllm.ErrProvider} {
		retry, backoff := p.ShouldRetry(ctx, 0, err)
		if !retry {
			t.Errorf("expected retry=true for %v, got false", err)
		}
		if backoff <= 0 {
			t.Errorf("expected positive backoff for %v, got %v", err, backoff)
		}
	}
}

func TestDefaultRetryPolicyNonRetryable(t *testing.T) {
	p := DefaultRetryPolicy{}
	ctx := context.Background()

	otherErr := errors.New("other")
	retry, backoff := p.ShouldRetry(ctx, 0, otherErr)
	if retry {
		t.Errorf("expected retry=false for non-retryable error, got true")
	}
	if backoff != 0 {
		t.Errorf("expected zero backoff for non-retryable error, got %v", backoff)
	}

	// nil error
	retry, backoff = p.ShouldRetry(ctx, 0, nil)
	if retry {
		t.Errorf("expected retry=false for nil error, got true")
	}
	if backoff != 0 {
		t.Errorf("expected zero backoff for nil error, got %v", backoff)
	}
}

func TestDefaultRetryPolicyBackoffIncreases(t *testing.T) {
	// Base at attempt 0 = 100ms, max with 25% jitter = 125ms.
	// Base at attempt 1 = 200ms, min with 25% jitter = 175ms.
	// So attempt1_min (175ms) > attempt0_max (125ms), making this deterministic.
	p := DefaultRetryPolicy{}
	ctx := context.Background()

	const iterations = 100
	for i := 0; i < iterations; i++ {
		_, b0 := p.ShouldRetry(ctx, 0, anyllm.ErrRateLimit)
		_, b1 := p.ShouldRetry(ctx, 1, anyllm.ErrRateLimit)

		maxB0 := 100*time.Millisecond + 25*time.Millisecond // base + 25% max jitter
		minB1 := 200 * time.Millisecond                     // base only, jitter >= 0

		if b0 > maxB0 {
			t.Errorf("attempt 0 backoff %v exceeds expected max %v", b0, maxB0)
		}
		if b1 < minB1 {
			t.Errorf("attempt 1 backoff %v below expected min %v", b1, minB1)
		}
	}
}
