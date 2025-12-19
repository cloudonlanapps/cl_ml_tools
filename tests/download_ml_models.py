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

print("=" * 70)
print("Downloading ML models for cl_ml_tools")
print("=" * 70)
print()

# Download CLIP model
print("1. Downloading CLIP (MobileCLIP-S2) model...")
try:
    from cl_ml_tools.plugins.clip_embedding.algo.clip_embedder import ClipEmbedder

    embedder = ClipEmbedder()
    print("   ✓ CLIP model downloaded")
except Exception as e:
    print(f"   ✗ Failed to download CLIP model: {e}")

print()

# Download DINO model
print("2. Downloading DINO (DINOv2) model...")
try:
    from cl_ml_tools.plugins.dino_embedding.algo.dino_embedder import DinoEmbedder

    embedder = DinoEmbedder()
    print("   ✓ DINO model downloaded")
except Exception as e:
    print(f"   ✗ Failed to download DINO model: {e}")

print()

# Download Face Detection model
print("3. Downloading Face Detection (MediaPipe) model...")
try:
    from cl_ml_tools.plugins.face_detection.algo.face_detector import FaceDetector

    detector = FaceDetector()
    print("   ✓ Face Detection model downloaded")
except Exception as e:
    print(f"   ✗ Failed to download Face Detection model: {e}")

print()

# Download Face Embedding model
print("4. Downloading Face Embedding (ArcFace) model...")
try:
    from cl_ml_tools.plugins.face_embedding.algo.face_embedder import FaceEmbedder

    embedder = FaceEmbedder()
    print("   ✓ Face Embedding model downloaded")
except Exception as e:
    print(f"   ✗ Failed to download Face Embedding model: {e}")

print()
print("=" * 70)
print("✓ ML model download complete!")
print("=" * 70)
print()

# Check cache directory
cache_dir = Path.home() / ".cache" / "cl_ml_tools" / "models"
if cache_dir.exists():
    models = list(cache_dir.glob("*.onnx"))
    print(f"Models cached in: {cache_dir}")
    print(f"Total models: {len(models)}")
    for model in models:
        size_mb = model.stat().st_size / (1024 * 1024)
        print(f"  - {model.name} ({size_mb:.1f} MB)")
