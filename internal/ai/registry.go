package ai

// Built-in model registry (SPEC v1 §4.6): stable model ids → on-device ONNX
// files plus the metadata GET /api/ai/models reports. Entries must stay
// within a decoder family the pre/post-processing implements (today:
// NanoDet GFL); see docs before adding a new family.

import (
	"errors"
	"os"
)

// ModelSpec is one entry of the built-in registry.
type ModelSpec struct {
	ID     string // registry id — the "model" value of SPEC §4.6
	Family string // decoder family; selects the pre/post-processing pair
	Input  int    // square input size in pixels (informational)
	Path   string // on-device ONNX file
}

// Registry lists the models this build can activate.
var Registry = []ModelSpec{
	{"nanodet-plus-m-320", "nanodet", 320, "/var/lib/mibee-eye/models/nanodet-m.onnx"},
	{"nanodet-plus-m-416", "nanodet", 416, "/var/lib/mibee-eye/models/nanodet-m-416.onnx"},
	{"yolox-nano-416", "yolox", 416, "/var/lib/mibee-eye/models/yolox-nano.onnx"},
}

// family selects the pre/post-processing pair for a model.
type family int

const (
	familyNanoDet family = iota
	familyYolox
)

// familyOf maps a registry family string; unknown values fall back to
// NanoDet (the escape hatch predates families).
func familyOf(s string) family {
	if s == "yolox" {
		return familyYolox
	}
	return familyNanoDet
}

// DefaultModelPath is the registry default for model_path overrides.
const DefaultModelPath = "/var/lib/mibee-eye/models/nanodet-m.onnx"

// Activation errors surfaced by ActivateModel (mapped to HTTP 404/409).
var (
	ErrUnknownModel     = errors.New("unknown model id")
	ErrModelUnavailable = errors.New("model file not available")
)

// ActiveModel is what a configuration resolves to.
type ActiveModel struct {
	ID     string
	Path   string
	Family string
	Input  int
	Custom bool
}

// Find looks up a registry entry by id.
func Find(id string) (ModelSpec, bool) {
	for _, m := range Registry {
		if m.ID == id {
			return m, true
		}
	}
	return ModelSpec{}, false
}

// Available reports whether the model file exists on this device (drives
// the "available" field of GET /api/ai/models).
func Available(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// Resolve maps (model id, model path) to the model that should be loaded.
// A model_path differing from DefaultModelPath is a custom deployment
// override and wins over the registry id. Returns ok=false for an unknown
// id with the default path (callers treat that as invalid configuration).
func Resolve(model, modelPath string) (ActiveModel, bool) {
	if modelPath != "" && modelPath != DefaultModelPath {
		return ActiveModel{
			ID:     "custom",
			Path:   modelPath,
			Family: "nanodet", // the decode path only implements NanoDet
			Custom: true,
		}, true
	}
	if model == "" {
		model = "nanodet-plus-m-320"
	}
	spec, ok := Find(model)
	if !ok {
		return ActiveModel{}, false
	}
	return ActiveModel{ID: spec.ID, Path: spec.Path, Family: spec.Family, Input: spec.Input}, true
}
