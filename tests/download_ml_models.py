#!/usr/bin/env python3
"""Download all ML models required for tests.

This script initializes each ML model-based plugin to trigger model downloads.
Models are cached in ~/.cache/cl_ml_tools/models/ for future use.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


# Download CLIP model
try:
    from cl_ml_tools.plugins.clip_embedding.algo.clip_embedder import ClipEmbedder

    embedder = ClipEmbedder()
except Exception:
    pass


# Download DINO model
try:
    from cl_ml_tools.plugins.dino_embedding.algo.dino_embedder import DinoEmbedder

    embedder = DinoEmbedder()
except Exception:
    pass


# Download Face Detection model
try:
    from cl_ml_tools.plugins.face_detection.algo.face_detector import FaceDetector

    detector = FaceDetector()
except Exception:
    pass


# Download Face Embedding model
try:
    from cl_ml_tools.plugins.face_embedding.algo.face_embedder import FaceEmbedder

    embedder = FaceEmbedder()
except Exception:
    pass


# Check cache directory
cache_dir = Path.home() / ".cache" / "cl_ml_tools" / "models"
if cache_dir.exists():
    models = list(cache_dir.glob("*.onnx"))
    for model in models:
        size_mb = model.stat().st_size / (1024 * 1024)
