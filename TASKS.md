# Task Tracking: cl_ml_tools Feature Expansion

**Current Phase**: Ready for Phase 2 - Face Recognition Infrastructure
**Last Updated**: 2025-12-16 12:00 PM
**Overall Progress**: 2/6 phases complete (Phase 0 ✓, Phase 1 ✓)

---

## Task Status Legend
- `[ ]` TODO - Not started
- `[~]` IN PROGRESS - Currently working on
- `[✓]` DONE - Completed

---

## Phase 0: Foundation & Tooling

### Workspace Artifacts
- `[✓]` Create `ImplementationPlan 1.md` in workspace root
- `[✓]` Create `TASKS.md` in workspace root

### Configuration & Tooling
- `[✓]` Configure pytest-cov in `pyproject.toml`
  - `[✓]` Add coverage thresholds (>85%)
  - `[✓]` Add HTML coverage report configuration
  - `[✓]` Add coverage exclusion patterns
- `[✓]` Create `INTERNALS.md` with standardized format
- `[✓]` Add ONNX runtime dependencies to `pyproject.toml`
  - `[✓]` Add `onnx>=1.15.0` to dependencies
  - `[✓]` Add `onnxruntime>=1.16.0` to dependencies
  - `[✓]` Add `numpy>=1.24.0` to dependencies
  - `[✓]` Add `httpx>=0.24` to dependencies (for model downloads)
- `[✓]` Create model download utility (`src/cl_ml_tools/utils/model_downloader.py`)
- `[✓]` Install new dependencies with `uv sync --all-extras`

**Note**: Skipped `.ai_context` folder (using TASKS.md instead). Skipped Git LFS (using download-on-demand strategy for models).

**Phase 0 Status**: ✅ COMPLETE - All tooling configured, dependencies resolved

---

## Phase 1: Exif Plugin Refactor

### Plugin Structure
- `[✓]` Create `src/cl_ml_tools/plugins/exif/schema.py`
  - `[✓]` Define `ExifParams` extending `BaseJobParams`
  - `[✓]` Define `ExifMetadata` Pydantic model with typed fields
  - `[✓]` Add raw dict field for complete metadata
- `[✓]` Create `src/cl_ml_tools/plugins/exif/task.py`
  - `[✓]` Define `ExifTask` extending `ComputeModule[ExifParams]`
  - `[✓]` Implement async `execute()` method
  - `[✓]` Add robust error handling (corrupt EXIF, unsupported formats)
  - `[✓]` Add logging throughout
- `[✓]` Create `src/cl_ml_tools/plugins/exif/routes.py`
  - `[✓]` Implement `create_router()` factory function
  - `[✓]` Add FastAPI endpoint for exif extraction
- `[✓]` Create `src/cl_ml_tools/plugins/exif/__init__.py`
  - `[✓]` Export `ExifTask`, `ExifParams`, and `ExifMetadata`

### Algorithm Refactor
- `[✓]` Refactor `src/cl_ml_tools/plugins/exif/algo/exif_tool_wrapper.py`
  - `[✓]` Replace print() with proper logging
  - `[✓]` Improve error handling (return defaults instead of exceptions)
  - `[✓]` Add type hints
  - `[✓]` Add docstrings
  - `[✓]` Add `extract_metadata_all()` method

### Testing
- `[✓]` Create `tests/test_exif_plugin.py`
  - `[✓]` Add schema validation tests (6 tests)
  - `[✓]` Add algorithm unit tests (6 tests)
  - `[✓]` Add task execution integration tests (5 tests)
  - `[✓]` Add error handling tests (2 tests)
  - `[✓]` All 19 tests passing ✅

### Documentation
- `[✓]` Create `src/cl_ml_tools/plugins/exif/README.md`
  - `[✓]` Document parameters
  - `[✓]` Document output schema
  - `[✓]` Add example usage
  - `[✓]` List dependencies (ExifTool)

### Plugin Registration
- `[✓]` Add exif plugin entry point to `pyproject.toml`
  - `[✓]` Add to `cl_ml_tools.tasks`
  - `[✓]` Add to `cl_ml_tools.routes`

**Phase 1 Status**: ✅ COMPLETE - Exif plugin fully implemented, tested (19/19 tests pass), and documented

---

## Phase 2: Face Recognition Infrastructure

### Module 1A: Face Detection Plugin

#### Model Preparation
- `[ ]` Research and source MediaPipe Face Detection ONNX model
- `[ ]` Document model source and version
- `[ ]` Test model loading and inference
- `[ ]` Define model storage location (Git LFS or download URL)

#### Plugin Structure
- `[ ]` Create `src/cl_ml_tools/plugins/face_detection/` directory structure
- `[ ]` Create `schema.py`
  - `[ ]` Define `FaceDetectionParams` with configurable threshold
  - `[ ]` Define `FaceDetectionOutput` with bbox list
  - `[ ]` Define `BoundingBox` model (normalized coordinates + confidence)
- `[ ]` Create `algo/face_detector.py`
  - `[ ]` Implement ONNX model loading
  - `[ ]` Implement preprocessing (resize, normalize)
  - `[ ]` Implement inference function
  - `[ ]` Implement NMS post-processing
- `[ ]` Create `task.py`
  - `[ ]` Define `FaceDetectionTask`
  - `[ ]` Implement batch processing
- `[ ]` Create `routes.py` with FastAPI endpoint
- `[ ]` Create `__init__.py` with exports

#### Testing
- `[ ]` Create `tests/test_face_detection_plugin.py`
  - `[ ]` Add schema validation tests
  - `[ ]` Add shape-based tests (input/output dimensions)
  - `[ ]` Add deterministic inference tests
  - `[ ]` Add golden vector tests (pre-computed bboxes)
  - `[ ]` Add performance benchmark tests

#### Documentation
- `[ ]` Create `src/cl_ml_tools/plugins/face_detection/README.md`
- `[ ]` Document model source and configuration
- `[ ]` Add example usage

### Module 1B: Face Embedding Plugin

#### Model Preparation
- `[ ]` Research and source MobileFaceNet (ArcFace) ONNX model
- `[ ]` Document model source and version
- `[ ]` Test model loading and inference
- `[ ]` Define model storage location

#### Plugin Structure
- `[ ]` Create `src/cl_ml_tools/plugins/face_embedding/` directory structure
- `[ ]` Create `schema.py`
  - `[ ]` Define `FaceEmbeddingParams`
  - `[ ]` Define `FaceEmbeddingOutput` (embedding + quality)
- `[ ]` Create `algo/face_aligner.py`
  - `[ ]` Implement 5-point landmark detection/alignment
  - `[ ]` Implement similarity transform
- `[ ]` Create `algo/face_embedder.py`
  - `[ ]` Implement ONNX model loading
  - `[ ]` Implement preprocessing
  - `[ ]` Implement inference function
  - `[ ]` Implement L2 normalization
- `[ ]` Create `algo/quality_scorer.py`
  - `[ ]` Implement blur/sharpness score calculation
- `[ ]` Create `task.py`
  - `[ ]` Define `FaceEmbeddingTask`
  - `[ ]` Implement face cropping from detection
  - `[ ]` Implement batch processing
- `[ ]` Create `routes.py` with FastAPI endpoint
- `[ ]` Create `__init__.py` with exports

#### Testing
- `[ ]` Create `tests/test_face_embedding_plugin.py`
  - `[ ]` Add schema validation tests
  - `[ ]` Add shape-based tests (128D or 512D output)
  - `[ ]` Add L2 normalization tests
  - `[ ]` Add golden vector tests (pre-computed embeddings)
  - `[ ]` Add deterministic inference tests
  - `[ ]` Add performance benchmark tests

#### Documentation
- `[ ]` Create `src/cl_ml_tools/plugins/face_embedding/README.md`
- `[ ]` Document model source and configuration
- `[ ]` Add example usage

### Plugin Registration
- `[ ]` Add face_detection and face_embedding to `pyproject.toml` entry points

**Phase 2 Completion Criteria**: Face plugins functional with FP32 models

---

## Phase 3: Embedding Infrastructure

### Module 2A: DINOv2 Embedding Plugin

#### Model Preparation
- `[ ]` Research and source DINOv2 ViT-S/14 ONNX model
- `[ ]` Document model source and version
- `[ ]` Test model loading and inference
- `[ ]` Define model storage location

#### Plugin Structure
- `[ ]` Create `src/cl_ml_tools/plugins/dino_embedding/` directory structure
- `[ ]` Create `schema.py`
  - `[ ]` Define `DinoEmbeddingParams`
  - `[ ]` Define `DinoEmbeddingOutput` (384D embedding)
- `[ ]` Create `algo/dino_preprocessor.py`
  - `[ ]` Implement image resize (224x224)
  - `[ ]` Implement ImageNet normalization
- `[ ]` Create `algo/dino_embedder.py`
  - `[ ]` Implement ONNX model loading
  - `[ ]` Implement inference function
  - `[ ]` Implement CLS token extraction
- `[ ]` Create `task.py`
  - `[ ]` Define `DinoEmbeddingTask`
  - `[ ]` Implement batch processing
- `[ ]` Create `routes.py` with FastAPI endpoint
- `[ ]` Create `__init__.py` with exports

#### Testing
- `[ ]` Create `tests/test_dino_embedding_plugin.py`
  - `[ ]` Add schema validation tests
  - `[ ]` Add shape-based tests (384D output)
  - `[ ]` Add golden vector tests
  - `[ ]` Add deterministic inference tests
  - `[ ]` Add similarity score validation tests

#### Documentation
- `[ ]` Create `src/cl_ml_tools/plugins/dino_embedding/README.md`

### Module 2B: MobileCLIP Embedding Plugin

#### Model Preparation
- `[ ]` Research and source MobileCLIP ONNX model (image encoder only)
- `[ ]` Document model source and version
- `[ ]` Test model loading and inference
- `[ ]` Define model storage location

#### Plugin Structure
- `[ ]` Create `src/cl_ml_tools/plugins/clip_embedding/` directory structure
- `[ ]` Create `schema.py`
  - `[ ]` Define `CLIPEmbeddingParams`
  - `[ ]` Define `CLIPEmbeddingOutput` (512D embedding)
- `[ ]` Create `algo/clip_preprocessor.py`
  - `[ ]` Implement CLIP-specific image preprocessing
  - `[ ]` Implement CLIP normalization
- `[ ]` Create `algo/clip_embedder.py`
  - `[ ]` Implement ONNX model loading
  - `[ ]` Implement inference function (image encoder only)
- `[ ]` Create `task.py`
  - `[ ]` Define `CLIPEmbeddingTask`
  - `[ ]` Implement batch processing
- `[ ]` Create `routes.py` with FastAPI endpoint
- `[ ]` Create `__init__.py` with exports

#### Testing
- `[ ]` Create `tests/test_clip_embedding_plugin.py`
  - `[ ]` Add schema validation tests
  - `[ ]` Add shape-based tests (512D output)
  - `[ ]` Add golden vector tests
  - `[ ]` Add deterministic inference tests
  - `[ ]` Add similarity score validation tests

#### Documentation
- `[ ]` Create `src/cl_ml_tools/plugins/clip_embedding/README.md`

### Plugin Registration
- `[ ]` Add dino_embedding and clip_embedding to `pyproject.toml` entry points

**Phase 3 Completion Criteria**: Embedding plugins produce deterministic outputs

---

## Phase 4: Documentation & Testing Completion

### Main Documentation
- `[ ]` Update `README.md`
  - `[ ]` Add new plugins to features list (5 new plugins)
  - `[ ]` Document model sourcing strategy
  - `[ ]` Add model cache locations section
  - `[ ]` Document offline behavior
  - `[ ]` Add licensing section (MediaPipe, DINOv2, CLIP)
  - `[ ]` Update "Adding New Plugins" section with `algo/` pattern
  - `[ ]` Document "resize → thumbnail" migration (temporary note)

### Contributing Guide
- `[ ]` Create `CONTRIBUTING.md`
  - `[ ]` Add plugin development walkthrough
  - `[ ]` Document `algo/` pattern requirements
  - `[ ]` Add testing guidelines
  - `[ ]` Add code style guidelines

### Utility Testing
- `[ ]` Create `tests/test_random_media_generator.py`
  - `[ ]` Test ImageGenerator
  - `[ ]` Test VideoGenerator
  - `[ ]` Test FrameGenerator
  - `[ ]` Test SceneGenerator
- `[ ]` Create `tests/test_media_types.py`
  - `[ ]` Test media type detection
  - `[ ]` Test MIME type handling
- `[ ]` Create `tests/test_timestamp.py`
  - `[ ]` Test timestamp utilities

### Plugin Testing Completion
- `[ ]` Create `tests/test_image_conversion_plugin.py`
  - `[ ]` Add schema validation tests
  - `[ ]` Add image conversion algorithm tests
  - `[ ]` Add task execution tests
- `[ ]` Update `tests/test_hash_plugin.py`
  - `[ ]` Replace hardcoded test media with random_media_generator
  - `[ ]` Ensure no skipped tests

### Coverage Verification
- `[ ]` Run pytest with coverage report
- `[ ]` Verify >85% overall code coverage
- `[ ]` Document any uncovered code in INTERNALS.md with justification

**Phase 4 Completion Criteria**: >85% code coverage, all docs complete

---

## Phase 5: Quality Assurance & Production Readiness

### Linting & Type Checking
- `[ ]` Run `ruff check src/` and fix all errors
- `[ ]` Run `ruff check tests/` and fix all errors
- `[ ]` Run `basedpyright src/` and fix all type errors
- `[ ]` Run `basedpyright tests/` and fix all type errors
- `[ ]` Document unavoidable warnings in INTERNALS.md

### Testing & Coverage
- `[ ]` Run full test suite: `pytest tests/`
- `[ ]` Generate coverage report: `pytest --cov=src/cl_ml_tools --cov-report=html`
- `[ ]` Review HTML coverage report
- `[ ]` Address any coverage gaps

### Performance Benchmarking
- `[ ]` Create benchmark script for Face Detection + Embedding pipeline
- `[ ]` Run benchmarks on CPU (FP32 baseline)
- `[ ]` Document FPS performance
- `[ ]` Compare against Phase 2 target (>5 FPS on RPi 4)

### Integration Testing
- `[ ]` Test all 9 plugins end-to-end via routes
  - `[ ]` hash
  - `[ ]` media_thumbnail
  - `[ ]` image_conversion
  - `[ ]` hls_streaming
  - `[ ]` exif
  - `[ ]` face_detection
  - `[ ]` face_embedding
  - `[ ]` dino_embedding
  - `[ ]` clip_embedding
- `[ ]` Test job queue processing with multiple concurrent jobs
- `[ ]` Test error scenarios
  - `[ ]` Missing ONNX models
  - `[ ]` Corrupt input files
  - `[ ]` Invalid parameters

### Production Documentation
- `[ ]` Create deployment guide
  - `[ ]` Document model installation process
  - `[ ]` List system requirements (CPU, RAM, disk)
  - `[ ]` Add deployment checklist
  - `[ ]` Add troubleshooting section
  - `[ ]` Document offline setup
- `[ ]` Update INTERNALS.md with final known issues

**Phase 5 Completion Criteria**: Production-ready, clean linter/type checker, deployment guide ready

---

## Phase 6: Model Optimization (Optional - Future)

### Quantization Pipeline
- `[ ]` Research onnxruntime quantization API
- `[ ]` Create representative calibration dataset
- `[ ]` Quantize MediaPipe Face Detection to INT8
- `[ ]` Quantize MobileFaceNet to INT8
- `[ ]` Quantize DINOv2 to INT8
- `[ ]` Quantize MobileCLIP to INT8

### Validation & Testing
- `[ ]` Validate INT8 accuracy vs FP32 (acceptable drift)
- `[ ]` Update tests to support both FP32 and INT8 models
- `[ ]` Run full test suite with INT8 models

### Performance Re-benchmarking
- `[ ]` Run benchmarks with INT8 models on CPU
- `[ ]` Document speedup vs FP32
- `[ ]` Verify >5 FPS target on RPi 4 achieved

### Documentation
- `[ ]` Document quantization process
- `[ ]` Create quantization scripts
- `[ ]` Update deployment guide with INT8 model instructions

**Phase 6 Completion Criteria**: INT8 quantized models achieve >5 FPS target on RPi 4 CPU

---

## Summary

**Total Tasks**: ~200+
**Completed**: 2 (workspace artifacts)
**In Progress**: 0
**Remaining**: ~198+

**Next Action**: Begin Phase 0 - Foundation & Tooling
