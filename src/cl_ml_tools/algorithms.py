"""Public algorithm API for cl_ml_tools.

This module exports core algorithm classes and functions for direct use
in customer applications without the FastAPI job infrastructure.

Example:
    Basic usage for image processing::

        from cl_ml_tools.algorithms import image_thumbnail, ClipEmbedder

        # Generate thumbnail
        image_thumbnail(
            input_path="photo.jpg",
            output_path="thumb.jpg",
            width=256,
            height=256
        )

        # Generate CLIP embedding
        embedder = ClipEmbedder()
        embedding = embedder.embed("photo.jpg")
        print(f"Embedding dimension: {embedding.shape[0]}")

    Face detection and analysis::

        from cl_ml_tools.algorithms import FaceDetector, FaceEmbedder

        # Detect faces
        detector = FaceDetector()
        faces = detector.detect("group_photo.jpg", confidence_threshold=0.7)
        print(f"Found {len(faces)} faces")

        # Generate face embeddings
        embedder = FaceEmbedder()
        for face in faces:
            embedding, quality = embedder.embed("cropped_face.jpg")

    File hashing::

        from cl_ml_tools.algorithms import get_md5_hexdigest, sha512hash_image

        # Simple MD5 hash
        from io import BytesIO
        with open("file.bin", "rb") as f:
            md5 = get_md5_hexdigest(BytesIO(f.read()))

        # Perceptual image hash
        with open("image.jpg", "rb") as f:
            img_hash, _ = sha512hash_image(BytesIO(f.read()))
"""

# Image Processing
# CLIP Embedding
from .plugins.clip_embedding.algo.clip_embedder import (
    ClipEmbedder,
)

# DINO Embedding
from .plugins.dino_embedding.algo.dino_embedder import (
    DinoEmbedder,
)

# EXIF
from .plugins.exif.algo.exif_tool_wrapper import (
    MetadataExtractor,
)

# Face Detection
from .plugins.face_detection.algo.face_detector import (
    FaceDetection,
    FaceDetector,
)

# Face Embedding
from .plugins.face_embedding.algo.face_embedder import (
    FaceEmbedder,
)

# Hashing
from .plugins.hash.algo.generic import (
    sha512hash_generic,
)
from .plugins.hash.algo.image import (
    sha512hash_image,
)
from .plugins.hash.algo.md5 import (
    get_md5_hexdigest,
)
from .plugins.hash.algo.video import (
    sha512hash_video2,
)

# HLS Streaming
from .plugins.hls_streaming.algo.hls_stream_generator import (
    HLSStreamGenerator,
    HLSVariant,
)
from .plugins.hls_streaming.algo.hls_validator import (
    HLSValidator,
    ValidationResult,
    validate_hls_output,
)
from .plugins.image_conversion.algo.image_convert import (
    image_convert,
)
from .plugins.media_thumbnail.algo.image_thumbnail import (
    image_thumbnail,
)
from .plugins.media_thumbnail.algo.video_thumbnail import (
    video_thumbnail,
)

# Utilities
from .utils.media_types import (
    MediaType,
    determine_media_type,
    determine_mime,
    get_extension_from_mime,
)
from .utils.random_media_generator import (
    RandomMediaGenerator,
)

__all__ = [
    # Image Processing
    "image_thumbnail",
    "video_thumbnail",
    "image_convert",
    # Hashing
    "get_md5_hexdigest",
    "sha512hash_image",
    "sha512hash_video2",
    "sha512hash_generic",
    # EXIF
    "MetadataExtractor",
    # Face Analysis
    "FaceDetector",
    "FaceDetection",
    "FaceEmbedder",
    # Embeddings
    "ClipEmbedder",
    "DinoEmbedder",
    # HLS Streaming
    "HLSStreamGenerator",
    "HLSVariant",
    "validate_hls_output",
    "HLSValidator",
    "ValidationResult",
    # Utilities
    "MediaType",
    "determine_mime",
    "determine_media_type",
    "get_extension_from_mime",
    "RandomMediaGenerator",
]
