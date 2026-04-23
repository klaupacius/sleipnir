package schema_test

import (
	"testing"

	"sleipnir.dev/sleipnir/internal/schema"
)

type simpleInput struct {
	Query string `json:"query" jsonschema:"required,description=Search query"`
	Limit int    `json:"limit"`
}

type chanInput struct {
	Ch chan int `json:"ch"`
}

func TestReflectSimpleStruct(t *testing.T) {
	m, err := schema.Reflect[simpleInput]()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if m["type"] != "object" {
		t.Errorf("expected type=object, got %v", m["type"])
	}
	props, ok := m["properties"].(map[string]any)
	if !ok {
		t.Fatalf("expected properties map, got %T", m["properties"])
	}
	if _, ok := props["query"]; !ok {
		t.Error("expected property 'query'")
	}
	if _, ok := props["limit"]; !ok {
		t.Error("expected property 'limit'")
	}
}

func TestReflectJsonNames(t *testing.T) {
	m, err := schema.Reflect[simpleInput]()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	props, ok := m["properties"].(map[string]any)
	if !ok {
		t.Fatalf("expected properties map, got %T", m["properties"])
	}
	if _, ok := props["query"]; !ok {
		t.Error("expected json name 'query', not 'Query'")
	}
	if _, ok := props["Query"]; ok {
		t.Error("property key should use json tag name, not field name")
	}
}

func TestReflectRequired(t *testing.T) {
	m, err := schema.Reflect[simpleInput]()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	req, ok := m["required"].([]any)
	if !ok {
		t.Fatalf("expected required array, got %T", m["required"])
	}
	found := false
	for _, v := range req {
		if v == "query" {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected 'query' in required array")
	}
}

func TestReflectNoRefs(t *testing.T) {
	m, err := schema.Reflect[simpleInput]()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := m["$defs"]; ok {
		t.Error("expected no $defs with DoNotReference=true")
	}
}

func TestReflectPanicRecovery(t *testing.T) {
	_, err := schema.Reflect[chanInput]()
	if err == nil {
		t.Error("expected error for channel type, got nil")
	}
}
