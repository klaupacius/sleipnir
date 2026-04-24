package schema

import (
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/invopop/jsonschema"
)

// Reflect generates a JSON Schema for type T as a map[string]any.
// T should be a struct with json: and jsonschema: tags.
// Returns an error if reflection fails or the reflector panics.
func Reflect[T any]() (m map[string]any, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("schema: reflection panicked for %s: %v",
				reflect.TypeFor[T]().String(), r)
		}
	}()

	r := &jsonschema.Reflector{DoNotReference: true}
	s := r.ReflectFromType(reflect.TypeFor[T]())

	data, err := json.Marshal(s)
	if err != nil {
		return nil, fmt.Errorf("schema: marshal failed: %w", err)
	}
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("schema: unmarshal failed: %w", err)
	}
	return m, nil
}
