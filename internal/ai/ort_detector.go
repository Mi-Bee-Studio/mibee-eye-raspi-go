//go:build ai

package ai

// ONNX Runtime detector behind the `ai` build tag (onnxruntime_go: CGO +
// dlopen of libonnxruntime.so, configured via [ai] onnx_lib_path).

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// preNMSConfidence trims obviously-empty grid points inside
// post-processing; the configured threshold filters the final results.
const preNMSConfidence float32 = 0.05

// ortInit guards ort.InitializeEnvironment: the ONNX Runtime global
// environment may only be initialized ONCE per process — a second call
// fails with "The onnxruntime has already been initialized", which would
// break every hot-swap after the first (SPEC §4.6 activate).
var (
	ortOnce     sync.Once
	ortInitErr  error
	ortLibTried string
)

func initOnnxRuntime(libPath string) error {
	ortOnce.Do(func() {
		ortLibTried = libPath
		if libPath != "" {
			ort.SetSharedLibraryPath(libPath)
		}
		ortInitErr = ort.InitializeEnvironment()
	})
	return ortInitErr
}

// OrtDetector runs NanoDet inference through ONNX Runtime.
type OrtDetector struct {
	modelPath   string
	session     *ort.DynamicAdvancedSession
	inputName   string
	outputName  string
	inputW, inputH uint32
	outputShape ort.Shape
	mu          sync.Mutex // session.Run is not safe for concurrent use
	closed      bool
}

// newOrtDetector loads the model and prepares the session. The shared
// library must be located before any onnxruntime call.
func newOrtDetector(opts Options) (Detector, error) {
	if err := initOnnxRuntime(opts.OnnxLibPath); err != nil {
		return nil, fmt.Errorf("ai: initialize onnxruntime: %w", err)
	}

	inputs, outputs, err := ort.GetInputOutputInfo(opts.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("ai: inspect model %s: %w", opts.ModelPath, err)
	}
	if len(inputs) == 0 || len(outputs) == 0 {
		return nil, fmt.Errorf("ai: model %s has no input/output tensors", opts.ModelPath)
	}
	inputInfo, outputInfo := inputs[0], outputs[0]
	inputDims, outputDims := inputInfo.Dimensions, outputInfo.Dimensions

	if len(inputDims) != 4 || inputDims[2] <= 0 || inputDims[3] <= 0 {
		return nil, fmt.Errorf("ai: model input shape must be static NCHW, got %v", inputDims)
	}
	inputH, inputW := inputDims[2], inputDims[3]
	if inputH != inputW {
		return nil, fmt.Errorf("ai: NanoDet expects a square input, got %dx%d", inputW, inputH)
	}
	// The expected point count follows the input size (gridLayout), so any
	// same-family export (320, 416, …) validates and decodes uniformly.
	expectedPoints := gridForInput(uint32(inputW)).points
	if len(outputDims) != 3 || outputDims[1] != int64(expectedPoints) || outputDims[2] != int64(numChannels) {
		return nil, fmt.Errorf("ai: model output shape must be [1,%d,%d], got %v (not a nanodet family model for input %d?)",
			expectedPoints, numChannels, outputDims, inputW)
	}
	outputShape := ort.Shape{outputDims[0], outputDims[1], outputDims[2]}

	session, err := ort.NewDynamicAdvancedSession(
		opts.ModelPath, []string{inputInfo.Name}, []string{outputInfo.Name}, nil)
	if err != nil {
		return nil, fmt.Errorf("ai: create session: %w", err)
	}

	return &OrtDetector{
		modelPath:   opts.ModelPath,
		session:     session,
		inputName:   inputInfo.Name,
		outputName:  outputInfo.Name,
		inputW:      uint32(inputW),
		inputH:      uint32(inputH),
		outputShape: outputShape,
	}, nil
}

// Detect runs preprocessing + inference + NanoDet GFL post-processing and
// scales bboxes to the video resolution.
func (d *OrtDetector) Detect(frame *Frame, videoW, videoH uint32) ([]Detection, error) {
	input, err := preprocessRGB8(frame.Data, frame.Width, frame.Height, d.inputW, d.inputH)
	if err != nil {
		return nil, err
	}

	inputTensor, err := ort.NewTensor(ort.NewShape(1, 3, int64(d.inputH), int64(d.inputW)), input)
	if err != nil {
		return nil, fmt.Errorf("ai: input tensor: %w", err)
	}
	defer inputTensor.Destroy()
	outputTensor, err := ort.NewEmptyTensor[float32](d.outputShape)
	if err != nil {
		return nil, fmt.Errorf("ai: output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	d.mu.Lock()
	if d.closed {
		d.mu.Unlock()
		return nil, fmt.Errorf("ai: session for %s was closed", d.modelPath)
	}
	err = d.session.Run([]ort.Value{inputTensor}, []ort.Value{outputTensor})
	d.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("ai: inference: %w", err)
	}

	grid := gridForInput(d.inputW)
	detections, err := postprocess(outputTensor.GetData(), grid, preNMSConfidence)
	if err != nil {
		return nil, err
	}
	return scaleDetectionsToFrame(detections, d.inputW, d.inputH, videoW, videoH), nil
}

// ModelName identifies the active model.
func (d *OrtDetector) ModelName() string { return d.modelPath }

// Close destroys the ONNX session. Hot-swaps call it on the replaced
// detector so its native (C++) memory is released immediately instead of
// waiting for a finalizer — the Pi 3B has no RAM to lend.
func (d *OrtDetector) Close() {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return
	}
	d.closed = true
	_ = d.session.Destroy()
}
