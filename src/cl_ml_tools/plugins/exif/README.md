# EXIF Metadata Extraction Plugin

Extracts EXIF metadata from images and other media files using ExifTool.

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | UploadFile | Yes | - | Media file to extract EXIF metadata from |
| `tags` | string | No | `""` | Comma-separated EXIF tags to extract (e.g., "Make,Model,DateTimeOriginal"). Leave empty to extract all available tags |
| `priority` | int | No | `5` | Job priority (0-10, higher = more urgent) |

## API Endpoint

```
POST /jobs/exif
```

### Example Request (Extract All Tags)

```bash
curl -X POST "http://localhost:8000/api/jobs/exif" \
  -F "file=@photo.jpg" \
  -F "tags="
```

### Example Request (Extract Specific Tags)

```bash
curl -X POST "http://localhost:8000/api/jobs/exif" \
  -F "file=@photo.jpg" \
  -F "tags=Make,Model,DateTimeOriginal,ISO,FNumber"
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
  "files": [
    {
      "file_path": "/path/to/input/photo.jpg",
      "status": "success",
      "metadata": {
        "make": "Canon",
        "model": "EOS R5",
        "date_time_original": "2024:01:15 10:30:00",
        "create_date": "2024:01:15 10:30:00",
        "image_width": 6000,
        "image_height": 4000,
        "orientation": 1,
        "iso": 400,
        "f_number": 2.8,
        "exposure_time": "1/125",
        "focal_length": 50.0,
        "gps_latitude": 37.7749,
        "gps_longitude": -122.4194,
        "gps_altitude": 100.5,
        "software": "Adobe Photoshop 2024",
        "raw_metadata": {
          "Make": "Canon",
          "Model": "EOS R5",
          ...
        }
      }
    }
  ],
  "total_files": 1,
  "tags_requested": "all"
}
```

## Output Schema

The `metadata` object contains typed fields for common EXIF tags:

| Field | Type | Description |
|-------|------|-------------|
| `make` | string \| null | Camera manufacturer (e.g., Canon, Nikon) |
| `model` | string \| null | Camera model (e.g., Canon EOS R5) |
| `date_time_original` | string \| null | Date/time when photo was taken |
| `create_date` | string \| null | File creation date |
| `image_width` | int \| null | Image width in pixels |
| `image_height` | int \| null | Image height in pixels |
| `orientation` | int \| null | Image orientation (1=normal, 3=180°, 6=90°CW, 8=270°CW) |
| `iso` | int \| null | ISO speed (e.g., 100, 400, 1600) |
| `f_number` | float \| null | F-stop/aperture (e.g., 2.8, 5.6) |
| `exposure_time` | string \| null | Shutter speed (e.g., 1/125, 1/1000) |
| `focal_length` | float \| null | Focal length in mm |
| `gps_latitude` | float \| null | GPS latitude in decimal degrees |
| `gps_longitude` | float \| null | GPS longitude in decimal degrees |
| `gps_altitude` | float \| null | GPS altitude in meters |
| `software` | string \| null | Software used to create/modify the image |
| `raw_metadata` | dict | Complete EXIF metadata dictionary from ExifTool |

## Error Handling

If a file has no EXIF metadata or an error occurs:

```json
{
  "file_path": "/path/to/file.jpg",
  "status": "no_metadata",  // or "error"
  "metadata": {
    "make": null,
    "model": null,
    ...
    "raw_metadata": {}
  },
  "error": "Error message (if status is 'error')"
}
```

## Supported File Types

ExifTool supports metadata extraction from:
- **Images**: JPEG, PNG, TIFF, RAW formats (CR2, NEF, ARW, etc.), HEIF/HEIC
- **Videos**: MP4, MOV, AVI, MKV, and many more
- **Audio**: MP3, FLAC, WAV (for embedded metadata)
- **Documents**: PDF (for metadata)

## Dependencies

Requires ExifTool to be installed on the system:

### macOS
```bash
brew install exiftool
```

### Ubuntu/Debian
```bash
sudo apt-get install libimage-exiftool-perl
```

### Windows
Download from: https://exiftool.org/

### Verify Installation
```bash
exiftool -ver
```

## Common EXIF Tags

Some commonly used EXIF tags you can extract:

**Camera Information:**
- `Make`, `Model`, `SerialNumber`, `LensModel`

**Capture Settings:**
- `ISO`, `FNumber`, `ExposureTime`, `FocalLength`, `Flash`

**Date/Time:**
- `DateTimeOriginal`, `CreateDate`, `ModifyDate`

**GPS Location:**
- `GPSLatitude`, `GPSLongitude`, `GPSAltitude`

**Image Properties:**
- `ImageWidth`, `ImageHeight`, `Orientation`, `ColorSpace`

**Software/Processing:**
- `Software`, `Artist`, `Copyright`

## Notes

- If `tags` parameter is empty or not provided, all available EXIF tags will be extracted
- The `raw_metadata` field always contains the complete metadata dictionary returned by ExifTool
- Tags are case-insensitive (e.g., "make" and "Make" are treated the same)
- Non-existent tags will return `null` values in the typed output
