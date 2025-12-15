# HLS Streaming Plugin

Converts video files to HLS (HTTP Live Streaming) format with adaptive bitrate support.

## Features

- **Adaptive Bitrate Streaming**: Generate multiple quality variants (720p, 480p, 240p, etc.)
- **Efficient Segmentation**: 2-second TS segments for smooth playback
- **Original Quality**: Optional original preservation without re-encoding
- **Validation**: Automatic output validation to ensure integrity
- **Smart Processing**: Incremental variant addition to existing streams

## Algorithm Details

The plugin uses FFmpeg to create HLS-compatible streams:

- **`algo/hls_stream_generator.py`**: Main conversion engine (HLSStreamGenerator, HLSVariant)
- **`algo/hls_validator.py`**: Output validation (HLSValidator)
- **`algo/ffmpeg_to_hls.py`**: Legacy FFmpeg wrapper

### Output Structure

```
output_dir/
├── adaptive.m3u8                    # Master playlist
├── adaptive-720p-3500.m3u8         # 720p variant playlist
├── adaptive-720p-3500-001.ts       # TS segments
├── adaptive-720p-3500-002.ts
├── ...
├── adaptive-480p-1500.m3u8         # 480p variant playlist
├── adaptive-480p-1500-001.ts
└── ...
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | UploadFile | Yes | - | Video file to convert |
| `variants` | JSON string | No | `[{720p,3500}, {480p,1500}]` | Quality variants to generate |
| `include_original` | bool | No | `false` | Preserve original quality |
| `priority` | int | No | `5` | Job priority (0-10) |

### Variant Configuration

Each variant is defined as:
```json
{
  "resolution": 720,  // Height in pixels
  "bitrate": 3500     // Target bitrate in kbps
}
```

## API Endpoint

```
POST /api/jobs/hls_streaming
```

### Example Request

```bash
curl -X POST "http://localhost:8000/api/jobs/hls_streaming" \
  -F "file=@video.mp4" \
  -F 'variants=[{"resolution":720,"bitrate":3500},{"resolution":480,"bitrate":1500}]' \
  -F "include_original=false"
```

### Example Response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

## Task Output

```json
{
  "files": [
    {
      "input_file": "/path/to/video.mp4",
      "output_dir": "/path/to/output",
      "master_playlist": "/path/to/output/adaptive.m3u8",
      "variants_generated": 2,
      "total_segments": 150,
      "include_original": false
    }
  ],
  "total_files": 1
}
```

## Dependencies

Requires:
- **FFmpeg** (system): Video encoding
- **ffprobe** (system): Metadata extraction
- **m3u8** (Python): Playlist parsing

Install with:
```bash
# Python dependencies
pip install cl_ml_tools[compute]

# System dependencies (Ubuntu/Debian)
sudo apt-get install ffmpeg

# System dependencies (macOS)
brew install ffmpeg
```

## Common Variants

| Resolution | Bitrate | Use Case |
|------------|---------|----------|
| 1080p | 5000 kbps | HD streaming, fast connections |
| 720p | 3500 kbps | HD streaming, standard quality |
| 480p | 1500 kbps | SD streaming, mobile/slow connections |
| 360p | 800 kbps | Low bandwidth, very slow connections |
| 240p | 400 kbps | Minimal bandwidth |
