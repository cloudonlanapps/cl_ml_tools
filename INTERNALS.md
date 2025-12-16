# Internal Documentation & Known Issues

This document tracks known technical issues, type checker warnings, and technical debt for `cl_ml_tools`.

## Type Checker Status

**Last checked**: 2025-12-16
**Tool**: basedpyright
**Summary**: 35 errors, 240 warnings

### Known Type Issues (Documented & Accepted)

#### 1. Missing ONNX Runtime Type Stubs

**Issue**: `reportMissingTypeStubs` for `onnxruntime` package
**Affected files**:
- `src/cl_ml_tools/plugins/clip_embedding/algo/clip_embedder.py`
- `src/cl_ml_tools/plugins/dino_embedding/algo/dino_embedder.py`
- `src/cl_ml_tools/plugins/face_detection/algo/face_detector.py`
- `src/cl_ml_tools/plugins/face_embedding/algo/face_embedder.py`

**Reason**: The `onnxruntime` package does not ship with type stubs, and no third-party stubs are available. This affects all ML plugins using ONNX models.

**Status**: **ACCEPTED** - External dependency limitation.

**Impact**: ~150 warnings related to unknown types from onnxruntime.

#### 2. Numpy Array Type Arguments

**Issue**: `reportMissingTypeArgument` for `np.ndarray`

**Reason**: Numpy's generic `ndarray` requires shape and dtype type parameters. Adding these makes signatures verbose without significant runtime benefit.

**Status**: **ACCEPTED** - Trade-off between type safety and code readability.

**Impact**: ~50 warnings

### Type Errors Requiring Fixes

See full list with `uv run basedpyright src/`

**Priority fixes**:
1. Missing error parameters in exception handling
2. Type incompatibility in ONNX inference results
3. Implicit string concatenation (5 instances)
4. Deprecated Union syntax (use `T | None` instead of `Optional[T]`)

## Code Coverage Status

**Last measured**: 2025-12-16
**Coverage**: 64.83% (349 tests passing)
**Target**: 85%

### Coverage Gaps

**High Priority (0% coverage)**:
- ML Plugin Routes (clip_embedding, dino_embedding, face_detection, face_embedding)
- exif plugin
- image_conversion plugin

**Medium Priority (<50%)**:
- Worker (21%)
- Model Downloader (21%)
- HLS Streaming (43-53%)

**Well-Tested (>80%)**:
- Utilities: timestamp (100%), media_types (98%), random_media_generator (76-100%)
- Hash plugin: 64-100%
- Media thumbnail: 83-93%
- MQTT: 81-88%

## Linter Status

**Tool**: ruff
**Status**: ✅ **PASSING**

## Production Readiness Checklist

- [x] Ruff linter passing
- [ ] Critical type errors fixed
- [ ] Coverage >85%
- [ ] Performance benchmarks
- [ ] Deployment guide

**Last updated**: 2025-12-16
