# Image Conversion Plugin

Converts images between different formats.

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | UploadFile | Yes | - | Image file to convert |
| `format` | string | Yes | - | Target format: `png`, `jpg`, `jpeg`, `webp`, `gif`, `bmp`, `tiff` |
| `quality` | int | No | `85` | Output quality for lossy formats (1-100) |
| `priority` | int | No | `5` | Job priority (0-10, higher = more urgent) |

## API Endpoint

```
POST /jobs/image_conversion
```

### Example Request

```bash
curl -X POST "http://localhost:8000/api/jobs/image_conversion" \
  -F "file=@photo.png" \
  -F "format=webp" \
  -F "quality=90"
```

### Example Response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

## Task Output

On successful completion, the job's `task_output` contains:

```json
{
  "processed_files": ["/path/to/output/converted_photo.webp"],
  "format": "webp",
  "quality": 90
}
```

## Format Notes

- **JPEG**: Does not support transparency. RGBA/P images are converted to RGB.
- **PNG**: Lossless, `quality` parameter is ignored.
- **WebP**: Supports both lossy and lossless (controlled by quality).
- **GIF**: Limited to 256 colors.

## Dependencies

Requires Pillow:

```bash
pip install cl_ml_tools[compute]
```
