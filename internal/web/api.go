package web

// SPEC v1 core endpoints: response envelope, cameras resource model,
// device status and the capability superset. Contract: ../mibee-webui/SPEC.md

import (
	"errors"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/Mi-Bee-Studio/mibee-eye-raspi/internal/ai"
	"gopkg.in/yaml.v3"
)

// errorCode maps an HTTP status to the SPEC §0 machine code.
func errorCode(status int) string {
	switch status {
	case http.StatusBadRequest:
		return "bad_request"
	case http.StatusUnauthorized:
		return "unauthorized"
	case http.StatusForbidden:
		return "forbidden"
	case http.StatusNotFound:
		return "not_found"
	case http.StatusConflict:
		return "conflict"
	case http.StatusTooManyRequests:
		return "rate_limited"
	case http.StatusNotImplemented:
		return "not_implemented"
	case http.StatusServiceUnavailable:
		return "setup_required"
	default:
		return "internal_error"
	}
}

// writeOK emits the success envelope {"ok":true,"data":…}.
func writeOK(w http.ResponseWriter, status int, data interface{}) {
	writeJSON(w, status, map[string]interface{}{"ok": true, "data": data})
}

// writeError emits the failure envelope
// {"ok":false,"error":"<machine code>","message":"<human text>"}.
func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]interface{}{
		"ok":      false,
		"error":   errorCode(status),
		"message": msg,
	})
}

// processStart backs the uptime fields.
var processStart = time.Now()

// handleHealth (GET /api/health, public): liveness probe (SPEC §1).
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeOK(w, http.StatusOK, map[string]interface{}{
		"status": "ok",
		"uptime": int(time.Since(processStart).Seconds()),
	})
}

// handleStatus (GET /api/status): SPEC §3 core fields + Go extras.
func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	oc := s.cfg.OnvifConfig
	resp := map[string]interface{}{
		"device_name": "MiBee Eye",
		"firmware":    s.cfg.Version,
		"uptime":      int(time.Since(processStart).Seconds()),
		"gb28181":     s.cfg.GB28181Config != nil && s.cfg.GB28181Config.Enabled,
	}
	if oc != nil {
		resp["model"] = oc.DeviceModel()
		resp["vendor"] = oc.DeviceManufacturer()
		resp["resolution"] = fmt.Sprintf("%dx%d", oc.CameraWidth(), oc.CameraHeight())
		resp["fps"] = oc.CameraFPS()
	}
	if s.cfg.CameraStatus != nil {
		resp["camera_alive"] = s.cfg.CameraStatus()
	}
	if s.cfg.RTSPStatus != nil {
		resp["rtsp"] = s.cfg.RTSPStatus()
	}
	writeOK(w, http.StatusOK, resp)
}

// handleCapabilities (GET /api/capabilities): SPEC §3.1 superset.
func (s *Server) handleCapabilities(w http.ResponseWriter, r *http.Request) {
	oc := s.cfg.OnvifConfig
	events := []string{"param_changed"}
	// Hot-swap is available exactly when a real detector is loaded: in
	// tag-less builds the factory fails, so AI is never active there.
	aiModels := s.cfg.AI != nil && s.cfg.AI.Active()
	if s.cfg.AI != nil && s.cfg.AI.Active() {
		events = append(events, "ai_detection")
	}
	if aiModels {
		events = append(events, "ai_model_changed")
	}
	caps := map[string]interface{}{
		"spec_version":      "1",
		"auth":              map[string]interface{}{"model": "session", "setup": true},
		"multi_camera":      false,
		"camera_management": false,
		"camera_control":    false,
		"imaging":           s.cfg.Params != nil,
		"ai":                s.cfg.AI != nil && s.cfg.AI.Active(),
		"ai_models":         aiModels,
		"ptz":               false,
		"hls":               true,
		"recording":         false,
		"devices":           false,
		"mjpeg":             false, // Go dialect A4: H.264 pipeline has no raw frames
		"mse":               s.cfg.AUHub != nil,
		"webrtc":            false,
		"events":            events,
		"config_apply": map[string]interface{}{
			"default":  "restart",
			"sections": map[string]string{"imaging": "immediate"},
			// SPEC §3.1: saving restart-sections self-restarts the service
			// immediately (附录A #10) — the frontend waits for recovery
			// instead of offering a manual restart entry.
			"auto": true,
		},
		"restart": true,
		"observability": map[string]interface{}{
			"metrics":  s.cfg.Metrics != nil,
			"logs":     true,
			"requests": true,
		},
	}
	device := map[string]interface{}{"name": "MiBee Eye"}
	if oc != nil {
		device["name"] = oc.DeviceName()
		device["model"] = oc.DeviceModel()
		device["vendor"] = oc.DeviceManufacturer()
	}
	caps["device"] = device
	writeOK(w, http.StatusOK, caps)
}

// handleDetections (GET /api/detections): SPEC v1 §4.6. Returns
// {"enabled":false} when AI is off/unavailable — never fabricated data.
func (s *Server) handleDetections(w http.ResponseWriter, r *http.Request) {
	if s.cfg.AI == nil || !s.cfg.AI.Active() {
		writeOK(w, http.StatusOK, map[string]interface{}{"enabled": false})
		return
	}
	writeOK(w, http.StatusOK, s.cfg.AI.Snapshot())
}

// handleAIModels (GET /api/ai/models): SPEC v1 §4.6 model registry. The
// "available" flag reflects the model file being present on this device;
// unavailable entries must not be activated.
func (s *Server) handleAIModels(w http.ResponseWriter, r *http.Request) {
	type modelEntry struct {
		ID        string `json:"id"`
		Family    string `json:"family"`
		Input     int    `json:"input"`
		Source    string `json:"source"`
		Available bool   `json:"available"`
	}
	models := make([]modelEntry, 0, len(ai.Registry))
	for _, spec := range ai.Registry {
		models = append(models, modelEntry{
			ID:        spec.ID,
			Family:    spec.Family,
			Input:     spec.Input,
			Source:    "builtin",
			Available: ai.Available(spec.Path),
		})
	}
	// A custom model_path override surfaces as its own (non-activatable)
	// entry so the UI can show what is running.
	if m, ok := s.resolveConfiguredModel(); ok && m.Custom {
		models = append(models, modelEntry{
			ID:        "custom",
			Family:    m.Family,
			Input:     m.Input,
			Source:    "custom",
			Available: ai.Available(m.Path),
		})
	}
	active := "unknown"
	if s.cfg.AI != nil && s.cfg.AI.Active() {
		active = s.cfg.AI.ModelID()
	} else if m, ok := s.resolveConfiguredModel(); ok {
		active = m.ID
	}
	writeOK(w, http.StatusOK, map[string]interface{}{
		"active": active,
		"models": models,
	})
}

// resolveConfiguredModel reads the ai section of the YAML config and maps
// it through the registry (mirror of the startup resolution). Falls back
// to the defaults when no config file is wired.
func (s *Server) resolveConfiguredModel() (ai.ActiveModel, bool) {
	var sec map[string]interface{}
	if s.cfg.ConfigPath != "" {
		if data, err := os.ReadFile(s.cfg.ConfigPath); err == nil {
			var cfg map[string]interface{}
			if err := yaml.Unmarshal(data, &cfg); err == nil {
				sec, _ = cfg["ai"].(map[string]interface{})
			}
		}
	}
	str := func(k string) string { v, _ := sec[k].(string); return v }
	return ai.Resolve(str("model"), str("model_path"))
}

// handleAIModelActivate (POST /api/ai/models/{id}/activate): SPEC v1 §4.6
// hot-switch. The service builds the new detector before touching the
// running slot (rollback by construction); on success the choice persists
// to ai.model and an ai_model_changed SSE event is broadcast.
func (s *Server) handleAIModelActivate(w http.ResponseWriter, r *http.Request) {
	if s.cfg.AI == nil || !s.cfg.AI.Active() {
		writeError(w, http.StatusNotImplemented, "AI module not running")
		return
	}
	id := r.PathValue("id")
	if err := s.cfg.AI.ActivateModel(id); err != nil {
		switch {
		case errors.Is(err, ai.ErrUnknownModel):
			writeError(w, http.StatusNotFound, err.Error())
		case errors.Is(err, ai.ErrModelUnavailable):
			writeError(w, http.StatusConflict, err.Error())
		default:
			writeError(w, http.StatusInternalServerError, err.Error())
		}
		return
	}
	if err := s.persistAIModel(id); err != nil {
		// The model is running; only the persisted default failed. Report
		// the error — the running state and the file diverge until the
		// next successful save.
		writeError(w, http.StatusInternalServerError,
			fmt.Sprintf("model activated but config not saved: %v", err))
		return
	}
	s.hub.broadcast("ai_model_changed", map[string]interface{}{
		"camera_id": "0",
		"model":     id,
	})
	writeOK(w, http.StatusOK, map[string]interface{}{
		"active":  id,
		"applied": "immediate",
	})
}

// persistAIModel merges the activated model id into the YAML ai section
// and atomically rewrites the file (same read-merge-write as
// handlePutConfig). A custom model_path is reset so the registry id wins
// on the next boot.
func (s *Server) persistAIModel(id string) error {
	if s.cfg.ConfigPath == "" {
		return fmt.Errorf("config path not configured")
	}
	data, err := os.ReadFile(s.cfg.ConfigPath)
	if err != nil {
		return fmt.Errorf("failed to read config: %w", err)
	}
	var cfg map[string]interface{}
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return fmt.Errorf("failed to parse config: %w", err)
	}
	aiSec, ok := cfg["ai"].(map[string]interface{})
	if !ok {
		aiSec = map[string]interface{}{}
		cfg["ai"] = aiSec
	}
	aiSec["model"] = id
	aiSec["model_path"] = ai.DefaultModelPath
	out, err := yaml.Marshal(cfg)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}
	return atomicWrite(s.cfg.ConfigPath, out)
}

// cameraDoc builds the single-camera document (SPEC §4).
func (s *Server) cameraDoc() map[string]interface{} {
	doc := map[string]interface{}{
		"id":          "0",
		"name":        "MiBee Eye",
		"camera_type": "csi",
		"status":      "online",
	}
	if oc := s.cfg.OnvifConfig; oc != nil {
		doc["name"] = oc.DeviceName()
		doc["resolution"] = fmt.Sprintf("%dx%d", oc.CameraWidth(), oc.CameraHeight())
		doc["fps"] = oc.CameraFPS()
		doc["rtsp_url"] = fmt.Sprintf("rtsp://self:%d/stream", oc.RTSPPort())
	}
	if s.cfg.CameraStatus != nil && !s.cfg.CameraStatus() {
		doc["status"] = "offline"
	}
	return doc
}

// handleCameras (GET /api/cameras): the single-element camera list.
func (s *Server) handleCameras(w http.ResponseWriter, r *http.Request) {
	writeOK(w, http.StatusOK, []interface{}{s.cameraDoc()})
}
