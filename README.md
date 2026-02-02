# cl_ml_tools

[![Version](https://img.shields.io/badge/version-0.2.1-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

**Master-worker media processing and machine learning toolkit** providing task plugins, worker runtime, and protocols for job persistence and file storage.

## Features

- **9 Production-Ready Plugins**: Image/video processing, ML embeddings, face analysis, and HLS streaming
- **FastAPI Integration**: Drop-in HTTP API with auto-discovery of plugins
- **Standalone Algorithm API**: Use ML models and processors directly without FastAPI
- **Async Job Queue**: Priority-based task scheduling with MQTT support
- **Hardware Acceleration**: Optimized for Raspberry Pi 5 + Hailo 8 AI Hat+
- **Type-Safe**: Full Pydantic validation and type hints throughout

## Plugins

1. **media_thumbnail** - Generate thumbnails from images and videos
2. **image_conversion** - Convert images between formats (PNG, JPG, WebP, etc.)
3. **hash** - Compute content hashes (SHA512, MD5, perceptual)
4. **exif** - Extract EXIF metadata from images
5. **clip_embedding** - MobileCLIP semantic image embeddings (512-dim)
6. **dino_embedding** - DINOv2 visual similarity embeddings (384-dim)
7. **face_detection** - YuNet face detection with bounding boxes
8. **face_embedding** - ArcFace embeddings for face recognition
9. **hls_streaming** - Multi-quality HLS video streaming conversion

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [FastAPI Routes Reference](#fastapi-routes-reference)
  - [Media Thumbnail](#1-media_thumbnail)
  - [Image Conversion](#2-image_conversion)
  - [Hash Computation](#3-hash)
  - [EXIF Extraction](#4-exif)
  - [CLIP Embedding](#5-clip_embedding)
  - [DINO Embedding](#6-dino_embedding)
  - [Face Detection](#7-face_detection)
  - [Face Embedding](#8-face_embedding)
  - [HLS Streaming](#9-hls_streaming)
- [Algorithm API Reference](#algorithm-api-reference)
- [Configuration](#configuration)
- [Testing](#testing)
- [License & Contributing](#license--contributing)

---

## Installation

### Basic Installation

```bash
pip install cl_ml_tools
```

### Development Installation

```bash
pip install cl_ml_tools[dev]
```

### System Dependencies

**ExifTool** (required for EXIF extraction):

```bash
# macOS
brew install exiftool

# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y libimage-exiftool-perl
```

**FFmpeg** (required for video processing):

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg
```

---

## Quick Start

### Standalone Algorithm Usage

Use algorithms directly without FastAPI infrastructure:

```python
from cl_ml_tools.algorithms import image_thumbnail, ClipEmbedder

# Generate a thumbnail
image_thumbnail(
    input_path="/path/to/photo.jpg",
    output_path="/path/to/thumb.jpg",
    width=256,
    height=256,
    maintain_aspect_ratio=True
)

# Generate CLIP embedding for semantic search
embedder = ClipEmbedder()
embedding = embedder.embed("/path/to/image.jpg")
print(f"Embedding shape: {embedding.shape}")  # (512,)
```

### FastAPI Server Setup

Create a production-ready API server:

```python
from fastapi import FastAPI
from cl_ml_tools import create_master_router, JobRepository, JobStorage

app = FastAPI(title="Media Processing API")

# Implement your persistence layer
class MyJobRepository(JobRepository):
    # Implement: create_job, get_job, update_job, list_jobs, delete_job
    pass

class MyFileStorage(JobStorage):
    # Implement: save_file, get_file, delete_file
    pass

repository = MyJobRepository()
file_storage = MyFileStorage()

# Optional: Authentication
async def get_current_user():
    return None  # Or return user from JWT/session

# Auto-discover and mount all 9 plugin routes
app.include_router(
    create_master_router(repository, file_storage, get_current_user),
    prefix="/api"
)

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## FastAPI Routes Reference

All routes return a `JobCreatedResponse` with structure:

```json
{
  "job_id": "uuid-string",
  "status": "queued",
  "task_type": "plugin_name",
  "created_at": "2025-12-18T10:30:00Z"
}
```

### 1. media_thumbnail

**POST** `/jobs/media_thumbnail`

Generate thumbnails from images or videos with automatic media type detection.

**Parameters:**
- `file` (required): Image or video file (multipart/form-data)
- `width` (required): Target width in pixels (integer > 0)
- `height` (required): Target height in pixels (integer > 0)
- `maintain_aspect_ratio` (optional): Maintain aspect ratio (boolean, default: false)
- `priority` (optional): Job priority 0-10 (integer, default: 5)

**Example:**

```bash
curl -X POST "http://localhost:8000/api/jobs/media_thumbnail" \
  -F "file=@photo.jpg" \
  -F "width=256" \
  -F "height=256" \
  -F "maintain_aspect_ratio=true" \
  -F "priority=5"
```

**Response:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "task_type": "media_thumbnail",
  "created_at": "2025-12-18T10:30:00Z"
}
```

**Output (when job completes):**
```json
{
  "media_type": "image",
  "output_path": "/path/to/thumbnail.jpg"
}
```

---

### 2. image_conversion

**POST** `/jobs/image_conversion`

Convert images between formats with quality control.

**Parameters:**
- `file` (required): Image file to convert
- `format` (required): Target format - one of: `png`, `jpg`, `jpeg`, `webp`, `gif`, `bmp`, `tiff`
- `quality` (optional): Output quality 1-100 for lossy formats (integer, default: 85)
- `priority` (optional): Job priority 0-10 (integer, default: 5)

**Example:**

```bash
curl -X POST "http://localhost:8000/api/jobs/image_conversion" \
  -F "file=@image.png" \
  -F "format=webp" \
  -F "quality=90" \
  -F "priority=5"
```

**Response:**

```json
{
  "job_id": "660e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "task_type": "image_conversion",
  "created_at": "2025-12-18T10:31:00Z"
}
```

**Output (when job completes):**
```json
{
  "output_path": "/path/to/converted.webp"
}
```

---

### 3. hash

**POST** `/jobs/hash`

Compute cryptographic or perceptual hashes for files.

**Parameters:**
- `file` (required): File to hash
- `algorithm` (optional): Hash algorithm - `sha512` or `md5` (default: `sha512`)
- `priority` (optional): Job priority 0-10 (integer, default: 5)

**Example:**

```bash
curl -X POST "http://localhost:8000/api/jobs/hash" \
  -F "file=@document.pdf" \
  -F "algorithm=sha512" \
  -F "priority=5"
```

**Response:**

```json
{
  "job_id": "770e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "task_type": "hash",
  "created_at": "2025-12-18T10:32:00Z"
}
```

**Output (when job completes):**
```json
{
  "media_type": "application/pdf",
  "hash": "abc123...",
  "algorithm": "sha512"
}
```

---

### 4. exif

**POST** `/jobs/exif`

Extract EXIF metadata from images using ExifTool.

**Parameters:**
- `file` (required): Image file
- `tags` (optional): Comma-separated EXIF tags to extract (empty string extracts all tags)
- `priority` (optional): Job priority 0-10 (integer, default: 5)

**Example:**

```bash
curl -X POST "http://localhost:8000/api/jobs/exif" \
  -F "file=@photo.jpg" \
  -F "tags=Make,Model,DateTimeOriginal,GPSLatitude,GPSLongitude" \
  -F "priority=5"
```

**Response:**

```json
{
  "job_id": "880e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "task_type": "exif",
  "created_at": "2025-12-18T10:33:00Z"
}
```

**Output (when job completes):**
```json
{
  "make": "Canon",
  "model": "EOS R5",
  "date_time_original": "2025:12:18 10:30:00",
  "create_date": "2025:12:18 10:30:00",
  "image_width": 4500,
  "image_height": 3000,
  "orientation": 1,
  "iso": 400,
  "f_number": 2.8,
  "exposure_time": "1/125",
  "focal_length": 50.0,
  "gps_latitude": 37.7749,
  "gps_longitude": -122.4194,
  "gps_altitude": 10.5,
  "software": "Adobe Lightroom",
  "raw_metadata": {
    "Make": "Canon",
    "Model": "EOS R5"
  }
}
```

---

### 5. clip_embedding

**POST** `/jobs/clip_embedding`

Generate MobileCLIP semantic embeddings for image similarity search.

**Parameters:**
- `file` (required): Image file
- `normalize` (optional): L2-normalize embedding (boolean, default: true)
- `priority` (optional): Job priority 0-10 (integer, default: 5)

**Example:**

```bash
curl -X POST "http://localhost:8000/api/jobs/clip_embedding" \
  -F "file=@image.jpg" \
  -F "normalize=true" \
  -F "priority=5"
```

**Response:**

```json
{
  "job_id": "990e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "task_type": "clip_embedding",
  "created_at": "2025-12-18T10:34:00Z"
}
```

**Output (when job completes):**
```json
{
  "embedding_dim": 512,
  "normalized": true,
  "output_path": "/path/to/clip_embedding.npy"
}
```

---

### 6. dino_embedding

**POST** `/jobs/dino_embedding`

Generate DINOv2 embeddings for visual similarity search.

**Parameters:**
- `file` (required): Image file
- `normalize` (optional): L2-normalize embedding (boolean, default: true)
- `priority` (optional): Job priority 0-10 (integer, default: 5)

**Example:**

```bash
curl -X POST "http://localhost:8000/api/jobs/dino_embedding" \
  -F "file=@image.jpg" \
  -F "normalize=true" \
  -F "priority=5"
```

**Response:**

```json
{
  "job_id": "aa0e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "task_type": "dino_embedding",
  "created_at": "2025-12-18T10:35:00Z"
}
```

**Output (when job completes):**
```json
{
  "embedding_dim": 384,
  "normalized": true,
  "output_path": "/path/to/dino_embedding.npy"
}
```

---

### 7. face_detection

**POST** `/jobs/face_detection`

Detect faces in images using YuNet with normalized bounding boxes.

**Parameters:**
- `file` (required): Image file
- `confidence_threshold` (optional): Minimum confidence 0.0-1.0 (float, default: 0.7)
- `nms_threshold` (optional): Non-maximum suppression threshold 0.0-1.0 (float, default: 0.4)
- `priority` (optional): Job priority 0-10 (integer, default: 5)

**Example:**

```bash
curl -X POST "http://localhost:8000/api/jobs/face_detection" \
  -F "file=@group_photo.jpg" \
  -F "confidence_threshold=0.7" \
  -F "nms_threshold=0.4" \
  -F "priority=5"
```

**Response:**

```json
{
  "job_id": "bb0e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "task_type": "face_detection",
  "created_at": "2025-12-18T10:36:00Z"
}
```

**Output (when job completes):**
```json
{
  "faces": [
    {
      "x1": 0.25,
      "y1": 0.30,
      "x2": 0.45,
      "y2": 0.60,
      "confidence": 0.98
    },
    {
      "x1": 0.60,
      "y1": 0.25,
      "x2": 0.80,
      "y2": 0.55,
      "confidence": 0.95
    }
  ],
  "num_faces": 2,
  "image_width": 1920,
  "image_height": 1080
}
```

---

### 8. face_embedding

**POST** `/jobs/face_embedding`

Generate ArcFace embeddings from cropped face images for face recognition.

**Parameters:**
- `file` (required): Cropped face image file
- `normalize` (optional): L2-normalize embedding (boolean, default: true)
- `priority` (optional): Job priority 0-10 (integer, default: 5)

**Example:**

```bash
curl -X POST "http://localhost:8000/api/jobs/face_embedding" \
  -F "file=@face_crop.jpg" \
  -F "normalize=true" \
  -F "priority=5"
```

**Response:**

```json
{
  "job_id": "cc0e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "task_type": "face_embedding",
  "created_at": "2025-12-18T10:37:00Z"
}
```

**Output (when job completes):**
```json
{
  "embedding_dim": 128,
  "normalized": true,
  "quality_score": 0.89,
  "output_path": "/path/to/face_embedding.npy"
}
```

---

### 9. hls_streaming

**POST** `/jobs/hls_streaming`

Convert videos to multi-quality HLS streaming format with adaptive bitrate.

**Parameters:**
- `file` (required): Video file
- `variants` (optional): JSON array of quality variants (string, see example below)
- `include_original` (optional): Include original quality (boolean, default: false)
- `priority` (optional): Job priority 0-10 (integer, default: 5)

**Default variants:**
```json
[
  {"resolution": 720, "bitrate": 3500},
  {"resolution": 480, "bitrate": 1500},
  {"resolution": 240, "bitrate": 800}
]
```

**Example:**

```bash
curl -X POST "http://localhost:8000/api/jobs/hls_streaming" \
  -F "file=@video.mp4" \
  -F 'variants=[{"resolution":1080,"bitrate":5000},{"resolution":720,"bitrate":3500},{"resolution":480,"bitrate":1500}]' \
  -F "include_original=false" \
  -F "priority=5"
```

**Response:**

```json
{
  "job_id": "dd0e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "task_type": "hls_streaming",
  "created_at": "2025-12-18T10:38:00Z"
}
```

**Output (when job completes):**
```json
{
  "master_playlist": "/path/to/output/master.m3u8",
  "variants_generated": 3,
  "total_segments": 45,
  "include_original": false
}
```

---

## Algorithm API Reference

Use algorithms directly without the FastAPI job infrastructure. All algorithms are exported from `cl_ml_tools.algorithms`.

### Image Processing

#### `image_thumbnail(input_path, output_path, width=None, height=None, maintain_aspect_ratio=True)`

Generate image thumbnail with optional aspect ratio preservation.

```python
from cl_ml_tools.algorithms import image_thumbnail

image_thumbnail(
    input_path="/path/to/image.jpg",
    output_path="/path/to/thumb.jpg",
    width=256,
    height=256,
    maintain_aspect_ratio=True
)
```

#### `video_thumbnail(input_path, output_path, width=None, height=None)`

Extract video frame as thumbnail.

```python
from cl_ml_tools.algorithms import video_thumbnail

video_thumbnail(
    input_path="/path/to/video.mp4",
    output_path="/path/to/thumb.jpg",
    width=256,
    height=256,
)
```

#### `image_convert(input_path, output_path, format="png", quality=85)`

Convert image to different format.

```python
from cl_ml_tools.algorithms import image_convert

image_convert(
    input_path="/path/to/image.png",
    output_path="/path/to/image.webp",
    format="webp",
    quality=90
)
```

---

### Hashing

#### `get_md5_hexdigest(file_like)`

Compute MD5 hash of file-like object.

```python
from cl_ml_tools.algorithms import get_md5_hexdigest
from io import BytesIO

with open("/path/to/file.bin", "rb") as f:
    md5 = get_md5_hexdigest(BytesIO(f.read()))
    print(f"MD5: {md5}")
```

#### `sha512hash_image(file_like)`

Compute perceptual SHA512 hash of image.

```python
from cl_ml_tools.algorithms import sha512hash_image

with open("/path/to/image.jpg", "rb") as f:
    img_hash, metadata = sha512hash_image(f)
    print(f"Hash: {img_hash}")
```

#### `sha512hash_video2(input_path)`

Compute SHA512 hash from video frames.

```python
from cl_ml_tools.algorithms import sha512hash_video2

video_hash = sha512hash_video2("/path/to/video.mp4")
print(f"Video hash: {video_hash}")
```

#### `sha512hash_generic(file_like)`

Generic SHA512 hash for any file.

```python
from cl_ml_tools.algorithms import sha512hash_generic

with open("/path/to/file.dat", "rb") as f:
    file_hash = sha512hash_generic(f)
```

---

### EXIF Metadata

#### `MetadataExtractor`

Extract EXIF metadata using ExifTool.

```python
from cl_ml_tools.algorithms import MetadataExtractor

extractor = MetadataExtractor()
metadata = extractor.extract_metadata("/path/to/image.jpg", tags=["Make", "Model", "DateTimeOriginal"])
print(metadata)
```

---

### Embeddings

#### `ClipEmbedder`

Generate MobileCLIP semantic embeddings (512-dim).

```python
from cl_ml_tools.algorithms import ClipEmbedder

embedder = ClipEmbedder()
embedding = embedder.embed("/path/to/image.jpg", normalize=True)
print(f"Shape: {embedding.shape}")  # (512,)
```

#### `DinoEmbedder`

Generate DINOv2 visual similarity embeddings (384-dim).

```python
from cl_ml_tools.algorithms import DinoEmbedder

embedder = DinoEmbedder()
embedding = embedder.embed("/path/to/image.jpg", normalize=True)
print(f"Shape: {embedding.shape}")  # (384,)
```

---

### Face Detection & Recognition

#### `FaceDetector`

Detect faces with YuNet.

```python
from cl_ml_tools.algorithms import FaceDetector

detector = FaceDetector()
faces = detector.detect(
    "/path/to/image.jpg",
    confidence_threshold=0.7,
    nms_threshold=0.4
)

for face in faces:
    print(f"Box: ({face.x1}, {face.y1}, {face.x2}, {face.y2})")
    print(f"Confidence: {face.confidence}")
```

#### `FaceEmbedder`

Generate ArcFace embeddings for face recognition.

```python
from cl_ml_tools.algorithms import FaceEmbedder

embedder = FaceEmbedder()
embedding, quality = embedder.embed("/path/to/face_crop.jpg", normalize=True)
print(f"Embedding shape: {embedding.shape}")  # (128,) or (512,)
print(f"Quality score: {quality}")
```

---

### HLS Streaming

#### `HLSStreamGenerator` and `HLSVariant`

Generate HLS streaming playlists with multiple quality variants.

```python
from cl_ml_tools.algorithms import HLSStreamGenerator, HLSVariant

variants = [
    HLSVariant(resolution=1080, bitrate=5000),
    HLSVariant(resolution=720, bitrate=3500),
    HLSVariant(resolution=480, bitrate=1500),
]

generator = HLSStreamGenerator()
output_dir = generator.generate(
    input_path="/path/to/video.mp4",
    output_dir="/path/to/output",
    variants=variants,
    include_original=False
)
print(f"HLS playlist: {output_dir}/master.m3u8")
```

#### `validate_hls_output(m3u8_file)`

Validate HLS output playlist structure.

```python
from cl_ml_tools.algorithms import validate_hls_output

error = validate_hls_output("/path/to/hls_output/master.m3u8")
if error:
    print(f"Validation failed: {error}")
```

---

### Utilities

#### `MediaType` enum and media detection

```python
from cl_ml_tools.algorithms import MediaType, determine_media_type, determine_mime

media_type = determine_media_type("/path/to/file.jpg")
print(media_type)  # MediaType.IMAGE

mime = determine_mime("/path/to/file.jpg")
print(mime)  # "image/jpeg"
```

#### `RandomMediaGenerator`

Generate random test media for testing.

```python
from cl_ml_tools.algorithms import RandomMediaGenerator

generator = RandomMediaGenerator()
image_path = generator.generate_image(width=1920, height=1080)
video_path = generator.generate_video(duration=10.0)
```

---

## Configuration

### Model Downloads

Models are automatically downloaded on first use to:

```
~/.cache/cl_ml_tools/models/
```

Downloaded models:
- **MobileCLIP S2** (~50 MB) - Semantic image embeddings
- **DINOv2 ViT-S** (~85 MB) - Visual similarity embeddings
- **YuNet** (~1 MB) - Face detection
- **ArcFace** (~130 MB) - Face recognition embeddings

### Environment Variables

Configure optional settings:

# MQTT model cache directory (not used for MQTT specifically but for models)
# export CL_ML_TOOLS_CACHE_DIR="/custom/cache/path"

### MQTT Setup (Optional)

For distributed worker deployments, configure MQTT:

```python
from cl_ml_tools import Worker, get_broadcaster

# Start MQTT broadcaster
broadcaster = get_broadcaster(mqtt_url="mqtt://broker.local:1883")

# Run worker with MQTT notifications
worker = Worker(repository, file_storage, broadcaster)
await worker.start()
```

---

---

## Testing

The package includes a comprehensive test suite covering all plugins and core functionality.

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=cl_ml_tools tests/
```

### Test Organization

- `tests/plugins/`: Plugin-specific algorithm and API tests
- `tests/utils/`: Utility function tests
- `tests/core/`: Worker runtime and protocol tests
- `tests/integration/`: End-to-end multi-plugin workflows

For more details on testing, see [tests/README.md](tests/README.md).

---

## License & Contributing

### License

**MIT License**

Copyright (c) 2025 Ananda Sarangaram

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Write tests** for new functionality (maintain 85%+ coverage)
3. **Follow code style**: Run `ruff` for linting
4. **Type hints**: Use full type annotations (checked with `basedpyright`)
5. **Documentation**: Update README and docstrings for API changes
6. **Commit messages**: Use conventional commits format

**Development setup:**

```bash
git clone https://github.com/yourusername/cl_ml_tools.git
cd cl_ml_tools
pip install -e .[dev]
pytest  # Ensure tests pass
```

**Adding a new plugin:**

1. Create plugin directory: `src/cl_ml_tools/plugins/my_plugin/`
2. Implement: `task.py`, `schema.py`, `routes.py`, `algo/`
3. Register in `pyproject.toml` entry points
4. Add tests in `tests/plugins/test_my_plugin.py`
5. Update this README

---

**Questions or issues?** Open an issue on GitHub or contact the maintainer.
