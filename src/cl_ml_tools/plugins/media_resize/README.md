# Image Resize Plugin

Resizes images to specified dimensions.

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | UploadFile | Yes | - | Image file to resize |
| `width` | int | Yes | - | Target width in pixels |
| `height` | int | Yes | - | Target height in pixels |
| `maintain_aspect_ratio` | bool | No | `false` | If true, maintains aspect ratio using thumbnail mode |
| `priority` | int | No | `5` | Job priority (0-10, higher = more urgent) |

## API Endpoint

```
POST /jobs/image_resize
```

### Example Request

```bash
curl -X POST "http://localhost:8000/api/jobs/image_resize" \
  -F "file=@photo.jpg" \
  -F "width=800" \
  -F "height=600" \
  -F "maintain_aspect_ratio=true"
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
  "processed_files": ["/path/to/output/resized_photo.jpg"],
  "dimensions": {"width": 800, "height": 600}
}
```

## Dependencies

Requires Pillow:

```bash
pip install cl_ml_tools[compute]
```
