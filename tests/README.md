# cl_ml_tools Test Suite

## Overview

Comprehensive test suite for cl_ml_tools with 85%+ code coverage. Tests cover all 9 plugins, utility modules, and integration flows. Uses real test media validated against MD5 manifests and includes both unit tests and end-to-end integration tests.

**Coverage Status:** 91.42%
**Test Count:** 359 tests 
**Test Media:** Validated via MANIFEST.md5
**External Dependencies:** FFmpeg, ExifTool (optional, can be excluded)

## Test Organization

```
tests/
├── conftest.py                 # Shared fixtures and pytest configuration
├── MANIFEST.md5                # MD5 checksums for test media (IN GIT)
├── setup_test_media.py         # Script to copy test media and generate manifest
├── generate_exif_test_media.py # Script to generate EXIF test images
├── test_media/                 # Real test assets (NOT IN GIT, only .keep)
│   ├── .keep                   # Placeholder for git
│   ├── images/                 # Test images
│   ├── videos/                 # Test videos
│   ├── audio/                  # Test audio files
│   └── exif_generated/         # Generated images with known EXIF
├── test_mqtt.py                # 54 existing MQTT tests (PRESERVED)
├── utils/
│   ├── test_media_types.py     # Media type detection tests
│   └── test_random_media_generator.py  # Random media generator tests
└── plugins/
    ├── test_clip_embedding.py
    ├── test_dino_embedding.py
    ├── test_face_detection.py
    ├── test_face_embedding.py
    ├── test_exif.py
    ├── test_hash.py
    ├── test_hls_streaming.py
    ├── test_image_conversion.py
    └── test_media_thumbnail.py
```

## Quick Start

### 1. Setup Test Media

Test media is NOT committed to git (too large). You must set it up locally:

```bash
# Copy test media from source directory and generate manifest
uv run python tests/setup_test_media.py

# Generate EXIF test images (requires exiftool)
uv run python tests/generate_exif_test_media.py
```

**Note:** `MANIFEST.md5` is committed to git, but `test_media/` is not (except `.keep` file).

### 2. Run All Tests

```bash
# Run all tests (requires FFmpeg, ExifTool, downloaded ML models)
uv run pytest

# With coverage report
uv run pytest --cov=src/cl_ml_tools --cov-report=html
```

### 3. Run Selective Tests

```bash
# Exclude external dependencies
uv run pytest -m "not requires_ffmpeg and not requires_exiftool"

# Exclude ML model tests
uv run pytest -m "not requires_models"

# Run only integration tests
uv run pytest -m "integration"

# Run single plugin tests
uv run pytest tests/plugins/test_hash.py
```

## Test Media Setup

### MANIFEST.md5 Structure

Located at `tests/MANIFEST.md5` (committed to git):

```
# Test Media Manifest
# Generated: 2025-12-18
# Source: ~/test_media

a1b2c3d4...  test_media/images/20210420_144043.jpg
b2c3d4e5...  test_media/images/20210420_225822.jpg
c3d4e5f6...  test_media/videos/sample_video.mp4
d4e5f6a1...  test_media/exif_generated/with_gps.jpg
```

**Important:** Paths are relative to `tests/` directory.

### Manifest Validation

Pytest automatically validates test media on startup via `conftest.py`:

```python
@pytest.fixture(scope="session", autouse=True)
def validate_test_media():
    """Validate test media exists and matches checksums."""
    # Checks test_media/ directory exists
    # Validates all files in MANIFEST.md5
    # Exits with error if validation fails
```

**If validation fails:**
```bash
Test media validation failed:
Missing: test_media/images/photo.jpg
Checksum mismatch: test_media/videos/video.mp4
  Expected: abc123...
  Actual:   def456...

Run: python tests/setup_test_media.py
```

### Regenerating Test Media

```bash
# Full reset (clears test_media/ and regenerates)
rm -rf tests/test_media
uv run python tests/setup_test_media.py
uv run python tests/generate_exif_test_media.py
```

## Pytest Markers

Tests use custom markers to indicate external dependencies.

### Available Markers

**@pytest.mark.requires_ffmpeg**
- Requires FFmpeg installed
- Used by: video_thumbnail, hls_streaming tests
- Exclude with: `pytest -m "not requires_ffmpeg"`

**@pytest.mark.requires_exiftool**
- Requires ExifTool installed
- Used by: exif extraction tests
- Exclude with: `pytest -m "not requires_exiftool"`

**@pytest.mark.requires_models**
- Requires ML models downloaded
- Used by: CLIP, DINO, face detection/embedding tests
- Models auto-download on first use (~100MB total)
- Exclude with: `pytest -m "not requires_models"`

**@pytest.mark.integration**
- Full integration tests (API → Worker → Completion)
- Tests entire job lifecycle
- Run only these with: `pytest -m "integration"`

### Example: Combining Markers

```bash
# Exclude all external dependencies
uv run pytest -m "not requires_ffmpeg and not requires_exiftool and not requires_models"

# Run only tests without FFmpeg requirement
uv run pytest -m "not requires_ffmpeg"

# Run integration tests that don't need FFmpeg
uv run pytest -m "integration and not requires_ffmpeg"
```

## Dependency Checking

**IMPORTANT:** Tests **FAIL** (not skip) if dependencies are missing.

This is intentional to ensure CI/CD catches missing dependencies:

```python
def pytest_runtest_setup(item):
    """Check dependencies before running tests."""
    if item.get_closest_marker("requires_ffmpeg"):
        if not shutil.which("ffmpeg"):
            pytest.fail(
                "FFmpeg not installed. "
                "Install: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)\n"
                "Or exclude with: pytest -m 'not requires_ffmpeg'"
            )
```

**Rationale:** Skipping silently can hide issues. Failing makes problems obvious.

## Shared Fixtures

Defined in `tests/conftest.py`:

### Session-Scoped Fixtures

**validate_test_media()**
- Validates MANIFEST.md5
- Runs once per test session
- Exits pytest if validation fails

### Function-Scoped Fixtures

**temp_output_dir(tmp_path)**
- Clean temporary directory for test outputs
- Unique per test
- Automatically cleaned up

**sample_image_path()**
- Returns path to a real test image from test_media/images/
- Skips test if no images available

**sample_video_path()**
- Returns path to a real test video from test_media/videos/
- Skips test if no videos available

**exif_test_image_path()**
- Returns path to image with known EXIF metadata
- From test_media/exif_generated/
- Skips if not generated

**synthetic_image(tmp_path)**
- Generates synthetic test image using PIL
- 800×600 with grid pattern
- No external file dependency

### Mock Service Fixtures

**job_repository()**
- In-memory JobRepository implementation
- No database required
- Fast, isolated

**file_storage(tmp_path, pytestconfig)**
- LocalFileStorage with configurable temp directory
- No external storage required
- Configurable via environment variable or pytest.ini

**worker(job_repository, file_storage)**
- Worker instance for integration tests
- Uses in-memory repository and storage

**api_client(job_repository, file_storage)**
- FastAPI TestClient
- Uses test dependencies
- For route testing

## Configuring Test Storage Location

By default, tests use pytest's temporary directory for job storage. You can override this for debugging or CI/CD environments.

### Configuration Priority

1. **TEST_STORAGE_DIR environment variable** (highest priority)
2. **pytest.ini test_storage_base_dir option**
3. **Default: tmp_path / "file_storage"** (pytest's temp dir)

### Environment Variable (Recommended for Debugging)

```bash
# Set storage location via environment variable
export TEST_STORAGE_DIR=/tmp/cl_ml_tools_test_storage
pytest tests/

# One-liner
TEST_STORAGE_DIR=/tmp/test_storage pytest tests/plugins/test_hash.py
```

### pytest.ini Configuration (Recommended for Team Settings)

In `pyproject.toml`:

```toml
[tool.pytest.ini_options]
# Uncomment and set your preferred path
test_storage_base_dir = "/tmp/cl_ml_tools_test_storage"
```

### Default Behavior

If neither is set, tests use `tmp_path / "file_storage"` (pytest's temporary directory, automatically cleaned up after tests).

### Use Cases

- **Debugging:** Set `TEST_STORAGE_DIR` to inspect test artifacts after test runs
- **CI/CD:** Configure via pytest.ini for consistent paths across team
- **Development:** Use default for automatic cleanup

## Integration Test Pattern

Each plugin includes an API → Worker integration test:

```python
@pytest.mark.integration
@pytest.mark.requires_models  # If ML model needed
def test_<plugin>_full_job_lifecycle(
    api_client, worker, job_repository, file_storage,
    sample_image_path, temp_output_dir
):
    """Test complete job lifecycle: API → Repository → Worker → Output."""

    # 1. Submit job via FastAPI route
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/<task_type>",
            files={"file": f},
            data={"priority": 5}
        )

    assert response.status_code == 200
    job_data = response.json()
    job_id = job_data["job_id"]

    # 2. Verify job in repository
    job_record = job_repository.get(job_id)
    assert job_record.status == "queued"

    # 3. Worker consumes job
    jobs_processed = worker.run_once()
    assert jobs_processed == 1

    # 4. Verify job completed
    job_record = job_repository.get(job_id)
    assert job_record.status == "completed"

    # 5. Verify output file exists and is valid
    # Plugin-specific validation here
```

## Plugin Test Summaries

### test_hash.py (~20 tests)
- MD5 hash consistency
- SHA-512 generic hashing
- SHA-512 image perceptual hashing
- SHA-512 video I-frame hashing
- Schema validation
- Task execution
- Route testing
- Integration test

### test_exif.py (~20 tests)
- ExifTool availability check
- Selective tag extraction
- Complete metadata extraction
- GPS coordinate extraction
- Camera settings parsing
- DateTime extraction
- Schema validation
- Integration test

### test_image_conversion.py (~15 tests)
- Format conversion (JPEG, PNG, WEBP)
- Quality settings
- RGBA → RGB conversion for JPEG
- Schema validation
- Task execution
- Integration test

### test_media_thumbnail.py (~20 tests)
- Image thumbnail with aspect ratio
- Image thumbnail forced dimensions
- Video thumbnail 4×4 grid
- Default dimensions handling
- Schema validation
- Integration tests

### test_clip_embedding.py (~20 tests)
- Model download
- Embedding generation
- L2 normalization
- Cosine similarity
- Schema validation
- Integration test
- Quality checks

### test_dino_embedding.py (~20 tests)
- Similar to CLIP but with DINOv2 model
- 384-dimensional embeddings
- CLS token extraction
- Integration test

### test_face_detection.py (~20 tests)
- Face bounding box detection
- Confidence threshold filtering
- NMS (Non-Maximum Suppression)
- Multiple faces handling
- Schema validation
- Integration test

### test_face_embedding.py (~20 tests)
- ArcFace embedding generation
- Quality score computation
- 512-dimensional embeddings
- Face verification workflow
- Integration test

### test_hls_streaming.py (~20 tests)
- Master playlist generation
- Multiple variant creation
- Segment file generation
- Directory validation
- Schema validation
- Integration test
- Requires FFmpeg

## Utility Test Summaries

### test_media_types.py (~25 tests)
- MediaType enum values
- MIME type detection
- File extension mapping
- determine_media_type() function
- URL validation
- Edge cases (unknown types, invalid inputs)

### test_random_media_generator.py (~25 tests)
- Random image generation
- Random video generation
- MIME type support
- Configuration validation
- Pydantic schema tests

## Running Tests in CI/CD

Example GitHub Actions workflow:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y ffmpeg exiftool

      - name: Install Python dependencies
        run: |
          pip install uv
          uv sync

      - name: Setup test media
        run: |
          python tests/setup_test_media.py
          python tests/generate_exif_test_media.py

      - name: Run tests with coverage
        run: |
          uv run pytest --cov=src/cl_ml_tools --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Troubleshooting

### "Test media directory not found"
```bash
# Run setup script
uv run python tests/setup_test_media.py
```

### "Test media manifest not found"
```bash
# Regenerate manifest
uv run python tests/setup_test_media.py
```

### "FFmpeg not installed"
```bash
# Install FFmpeg
brew install ffmpeg  # macOS
sudo apt-get install ffmpeg  # Linux

# Or exclude FFmpeg tests
uv run pytest -m "not requires_ffmpeg"
```

### "ExifTool not installed"
```bash
# Install ExifTool
brew install exiftool  # macOS
sudo apt-get install libimage-exiftool-perl  # Linux

# Or exclude ExifTool tests
uv run pytest -m "not requires_exiftool"
```

### "ML models not downloaded"
```bash
# Models auto-download on first use
# Just run the tests, they'll download (~100MB total)
uv run pytest tests/plugins/test_clip_embedding.py

# Or exclude model tests
uv run pytest -m "not requires_models"
```

### "Checksum mismatch"
```bash
# Test media files were modified or corrupted
# Regenerate test media
rm -rf tests/test_media
uv run python tests/setup_test_media.py
```

## Coverage Reports

### Generate HTML Coverage Report
```bash
uv run pytest --cov=src/cl_ml_tools --cov-report=html
open htmlcov/index.html
```

### Generate Terminal Coverage Report
```bash
uv run pytest --cov=src/cl_ml_tools --cov-report=term-missing
```

### Coverage by Module
```bash
uv run pytest --cov=src/cl_ml_tools --cov-report=term-missing --cov-branch
```

**Expected Coverage:**
- Overall: 85%+
- Plugins: 90%+ (high test coverage)
- Common modules: 80%+ (shared code)
- Utils: 90%+ (utility functions)

## Test Development Guidelines

### Writing New Tests

1. **Use existing fixtures:** Leverage conftest.py fixtures
2. **Mark dependencies:** Add appropriate `@pytest.mark.requires_*` decorators
3. **Follow pattern:** Use existing plugin tests as templates
4. **Integration tests:** Include at least one per plugin
5. **Validate outputs:** Check file existence, content, and schema

### Example Test Structure
```python
@pytest.mark.requires_ffmpeg
def test_video_thumbnail_basic(sample_video_path, temp_output_dir):
    """Test basic video thumbnail generation."""
    # Arrange
    output_path = temp_output_dir / "thumb.jpg"

    # Act
    result = video_thumbnail(
        input_path=sample_video_path,
        output_path=output_path,
        width=256,
        height=256
    )

    # Assert
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    # Additional validation...
```

## Additional Resources

- **Plugin READMEs:** See `src/cl_ml_tools/plugins/*/algo/README.md` for algorithm details
- **API Documentation:** See root `README.md` for API usage
- **Issue Tracker:** [GitHub Issues](https://github.com/your-org/cl_ml_tools/issues)

## Version History

- **v0.2.1:** Current test suite with 91.42% coverage
- **v0.2.0:** Initial comprehensive test suite

## Support

For test-related issues:
- Review this README
- Check `conftest.py` for available fixtures
- Run with `-v` flag for verbose output: `pytest -v`
- File issues on GitHub
