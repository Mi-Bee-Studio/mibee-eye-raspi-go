package ai

import "testing"

func TestResolveDefaultIsRegistry320(t *testing.T) {
	m, ok := Resolve("", DefaultModelPath)
	if !ok {
		t.Fatal("empty model id must default to the 320 entry")
	}
	if m.ID != "nanodet-plus-m-320" || m.Custom || m.Family != "nanodet" || m.Input != 320 {
		t.Fatalf("m = %+v", m)
	}
	if m.Path != "/var/lib/mibee-eye/models/nanodet-m.onnx" {
		t.Fatalf("path = %s", m.Path)
	}
}

func TestResolve416Entry(t *testing.T) {
	m, ok := Resolve("nanodet-plus-m-416", DefaultModelPath)
	if !ok {
		t.Fatal("registry id must resolve")
	}
	if m.Input != 416 || m.Path != "/var/lib/mibee-eye/models/nanodet-m-416.onnx" {
		t.Fatalf("m = %+v", m)
	}
}

func TestResolveCustomModelPathWins(t *testing.T) {
	m, ok := Resolve("nanodet-plus-m-320", "/opt/my-nanodet.onnx")
	if !ok {
		t.Fatal("custom path must resolve")
	}
	if !m.Custom || m.ID != "custom" || m.Path != "/opt/my-nanodet.onnx" {
		t.Fatalf("m = %+v", m)
	}
}

func TestResolveUnknownIDRejected(t *testing.T) {
	if _, ok := Resolve("yolo-9000", DefaultModelPath); ok {
		t.Fatal("unknown id must be rejected")
	}
}

func TestRegistryIDsUnique(t *testing.T) {
	seen := map[string]bool{}
	for _, m := range Registry {
		if seen[m.ID] {
			t.Fatalf("duplicate id %s", m.ID)
		}
		seen[m.ID] = true
	}
}
