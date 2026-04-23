package sleipnirtest

import anyllm "github.com/mozilla-ai/any-llm-go"

var _ anyllm.Provider = (*StubProvider)(nil)
