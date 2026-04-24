package retry

import (
	"context"
	"errors"
	"math/rand"
	"time"

	anyllm "github.com/mozilla-ai/any-llm-go"
	sleipnir "sleipnir.dev/sleipnir"
)

// DefaultRetryPolicy retries anyllm.ErrRateLimit and anyllm.ErrProvider with
// exponential backoff + 25% jitter. Stateless — the attempt count is passed in.
type DefaultRetryPolicy struct {
	sleipnir.BaseMiddleware
}

func (DefaultRetryPolicy) ShouldRetry(_ context.Context, attempt int, err error) (bool, time.Duration) {
	if !isRetryable(err) {
		return false, 0
	}
	base := time.Duration(1<<uint(attempt)) * 100 * time.Millisecond
	jitter := time.Duration(rand.Int63n(int64(base / 4)))
	return true, base + jitter
}

func isRetryable(err error) bool {
	return errors.Is(err, anyllm.ErrRateLimit) || errors.Is(err, anyllm.ErrProvider)
}

var _ sleipnir.RetryPolicy = DefaultRetryPolicy{}
