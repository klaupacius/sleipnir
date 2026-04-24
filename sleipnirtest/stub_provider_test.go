package sleipnirtest

import (
	"context"
	"regexp"
	"testing"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

var _ anyllm.Provider = (*StubProvider)(nil)

func TestStubProviderMatcherUsed(t *testing.T) {
	resp := TextResponse("matched")
	fallback := TextResponse("fallback")
	sp := NewStubProvider(t, fallback).
		WhenMessageMatches(regexp.MustCompile("hello"), resp)

	params := anyllm.CompletionParams{
		Messages: []anyllm.Message{
			{Role: anyllm.RoleUser, Content: "say hello there"},
		},
	}
	got, err := sp.Completion(context.Background(), params)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Choices[0].Message.Content != "matched" {
		t.Errorf("expected 'matched', got %q", got.Choices[0].Message.Content)
	}
}

func TestStubProviderMatcherFallthrough(t *testing.T) {
	matched := TextResponse("matched")
	fallback := TextResponse("fallback")
	sp := NewStubProvider(t, fallback).
		WhenMessageMatches(regexp.MustCompile("hello"), matched)

	params := anyllm.CompletionParams{
		Messages: []anyllm.Message{
			{Role: anyllm.RoleUser, Content: "goodbye world"},
		},
	}
	got, err := sp.Completion(context.Background(), params)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Choices[0].Message.Content != "fallback" {
		t.Errorf("expected 'fallback', got %q", got.Choices[0].Message.Content)
	}
}

func TestStubProviderMatcherOrder(t *testing.T) {
	first := TextResponse("first")
	second := TextResponse("second")
	sp := NewStubProvider(t).
		WhenMessageMatches(regexp.MustCompile("foo"), first).
		WhenMessageMatches(regexp.MustCompile("foo"), second)

	params := anyllm.CompletionParams{
		Messages: []anyllm.Message{
			{Role: anyllm.RoleUser, Content: "foo bar"},
		},
	}
	got, err := sp.Completion(context.Background(), params)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Choices[0].Message.Content != "first" {
		t.Errorf("expected 'first' (first matcher wins), got %q", got.Choices[0].Message.Content)
	}
}
