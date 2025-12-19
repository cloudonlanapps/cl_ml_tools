"""Unit and integration tests for image conversion plugin.

Tests schema validation, format conversion algorithms (JPEG↔PNG), task execution, routes, and full job lifecycle.
"""

from pathlib import Path

import pytest
from PIL import Image

from cl_ml_tools.plugins.image_conversion.algo.image_convert import (
    get_pil_format,
    image_convert,
)
from cl_ml_tools.plugins.image_conversion.schema import (
    ImageConversionOutput,
    ImageConversionParams,
)
from cl_ml_tools.plugins.image_conversion.task import ImageConversionTask

# ============================================================================
# SCHEMA TESTS
# ============================================================================


def test_image_conversion_params_schema_validation():
    """Test ImageConversionParams schema validates correctly."""
    params = ImageConversionParams(
        input_path="/path/to/input.jpg",
        output_path="output/converted.png",
        format="png",
        quality=90,
    )

    assert params.input_path == "/path/to/input.jpg"
    assert params.output_path == "output/converted.png"
    assert params.format == "png"
    assert params.quality == 90


def test_image_conversion_params_defaults():
    """Test ImageConversionParams has correct default values."""
    params = ImageConversionParams(
        input_path="/path/to/input.jpg",
        output_path="output/converted.png",
        format="png",
    )

    assert params.quality == 85


def test_image_conversion_params_quality_validation():
    """Test ImageConversionParams validates quality range."""
    # Valid quality
    params = ImageConversionParams(
        input_path="/path/to/input.jpg",
        output_path="output/converted.png",
        format="png",
        quality=95,
    )
    assert params.quality == 95

    # Invalid quality (too high)
    with pytest.raises(ValueError):
        _ = ImageConversionParams(
            input_path="/path/to/input.jpg",
            output_path="output/converted.png",
            format="png",
            quality=101,
        )

    # Invalid quality (too low)
    with pytest.raises(ValueError):
        _ = ImageConversionParams(
            input_path="/path/to/input.jpg",
            output_path="output/converted.png",
            format="png",
            quality=0,
        )


def test_image_conversion_params_format_validation():
    """Test ImageConversionParams validates format field."""
    # Valid formats
    for fmt in ["png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"]:
        params = ImageConversionParams(
            input_path="/path/to/input.jpg",
            output_path=f"output/converted.{fmt}",
            format=fmt,  # type: ignore
        )
        assert params.format == fmt

    # Invalid format
    with pytest.raises(ValueError):
        _ = ImageConversionParams(
            input_path="/path/to/input.jpg",
            output_path="output/converted.svg",
            format="svg",  # type: ignore
        )


def test_image_conversion_output_schema_validation():
    """Test ImageConversionOutput schema validates correctly."""
    output = ImageConversionOutput()

    assert isinstance(output, ImageConversionOutput)


# ============================================================================
# ALGORITHM TESTS - Format Helpers
# ============================================================================


def test_get_pil_format_jpg():
    """Test _get_pil_format converts jpg to JPEG."""
    assert get_pil_format("jpg") == "JPEG"
    assert get_pil_format("jpeg") == "JPEG"


def test_get_pil_format_png():
    """Test _get_pil_format converts png to PNG."""
    assert get_pil_format("png") == "PNG"


def test_get_pil_format_other_formats():
    """Test _get_pil_format handles various formats."""
    assert get_pil_format("webp") == "WEBP"
    assert get_pil_format("gif") == "GIF"
    assert get_pil_format("bmp") == "BMP"
    assert get_pil_format("tiff") == "TIFF"


def test_get_pil_format_case_insensitive():
    """Test _get_pil_format is case insensitive."""
    assert get_pil_format("PNG") == "PNG"
    assert get_pil_format("JpG") == "JPEG"


# ============================================================================
# ALGORITHM TESTS - Conversion
# ============================================================================


def test_image_convert_jpg_to_png(sample_image_path: Path, tmp_path: Path):
    """Test converting JPEG to PNG."""
    output_path = tmp_path / "output.png"

    result = image_convert(
        input_path=sample_image_path,
        output_path=output_path,
        format="png",
    )

    assert result == str(output_path)
    assert output_path.exists()

    # Verify it's a valid PNG
    with Image.open(output_path) as img:
        assert img.format == "PNG"


def test_image_convert_png_to_jpg(synthetic_image: Path, tmp_path: Path):
    """Test converting PNG to JPEG."""
    # First create a PNG from synthetic image
    png_path = tmp_path / "input.png"
    with Image.open(synthetic_image) as img:
        img.save(png_path, "PNG")

    output_path = tmp_path / "output.jpg"

    result = image_convert(
        input_path=png_path,
        output_path=output_path,
        format="jpg",
        quality=90,
    )

    assert result == str(output_path)
    assert output_path.exists()

    # Verify it's a valid JPEG
    with Image.open(output_path) as img:
        assert img.format == "JPEG"


def test_image_convert_with_quality(sample_image_path: Path, tmp_path: Path):
    """Test image conversion with quality parameter."""
    output_path = tmp_path / "output.jpg"

    _ = image_convert(
        input_path=sample_image_path,
        output_path=output_path,
        format="jpg",
        quality=50,
    )

    assert output_path.exists()

    # Lower quality should produce smaller file
    output_path_high = tmp_path / "output_high.jpg"
    _ = image_convert(
        input_path=sample_image_path,
        output_path=output_path_high,
        format="jpg",
        quality=95,
    )

    # High quality file should be larger
    assert output_path_high.stat().st_size > output_path.stat().st_size


def test_image_convert_rgba_to_jpg(tmp_path: Path):
    """Test converting RGBA image to JPEG (alpha channel removed)."""
    # Create RGBA image
    rgba_image = tmp_path / "input_rgba.png"
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    img.save(rgba_image, "PNG")

    output_path = tmp_path / "output.jpg"

    _ = image_convert(
        input_path=rgba_image,
        output_path=output_path,
        format="jpg",
    )

    assert output_path.exists()

    # Verify conversion to RGB
    with Image.open(output_path) as img:
        assert img.format == "JPEG"
        assert img.mode == "RGB"


def test_image_convert_file_not_found(tmp_path: Path):
    """Test image_convert raises FileNotFoundError for missing input."""
    output_path = tmp_path / "output.png"

    with pytest.raises(FileNotFoundError):
        image_convert(
            input_path="/nonexistent/file.jpg",
            output_path=output_path,
            format="png",
        )


def test_image_convert_output_directory_not_found(sample_image_path: Path, tmp_path: Path):
    """Test image_convert raises FileNotFoundError when output directory doesn't exist."""
    output_path = tmp_path / "nonexistent" / "dir" / "output.png"

    with pytest.raises(FileNotFoundError, match="Output directory does not exist"):
        image_convert(
            input_path=sample_image_path,
            output_path=output_path,
            format="png",
        )


def test_image_convert_webp_format(sample_image_path: Path, tmp_path: Path):
    """Test converting to WebP format."""
    output_path = tmp_path / "output.webp"

    image_convert(
        input_path=sample_image_path,
        output_path=output_path,
        format="webp",
        quality=80,
    )

    assert output_path.exists()

    with Image.open(output_path) as img:
        assert img.format == "WEBP"


# ============================================================================
# TASK TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_image_conversion_task_run_success(sample_image_path: Path, tmp_path: Path):
    """Test ImageConversionTask execution success."""
    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(sample_image_path.read_bytes())

    params = ImageConversionParams(
        input_path=str(input_path),
        output_path="output/converted.png",
        format="png",
        quality=90,
    )

    task = ImageConversionTask()
    job_id = "test-job-123"

    class MockStorage:
        def resolve_path(self, job_id: str, relative_path: str) -> Path:
            return tmp_path / job_id / relative_path

        def allocate_path(self, job_id: str, relative_path: str) -> str:
            output_path = tmp_path / "output" / "converted.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return str(output_path)

    storage = MockStorage()

    output = await task.run(job_id, params, storage)

    assert isinstance(output, ImageConversionOutput)

    # Verify output file
    output_file = tmp_path / "output" / "converted.png"
    assert output_file.exists()


@pytest.mark.asyncio
async def test_image_conversion_task_run_file_not_found(tmp_path: Path):
    """Test ImageConversionTask raises FileNotFoundError for missing input."""
    params = ImageConversionParams(
        input_path="/nonexistent/file.jpg",
        output_path="output/converted.png",
        format="png",
    )

    task = ImageConversionTask()
    job_id = "test-job-789"

    class MockStorage:
        def resolve_path(self, job_id: str, relative_path: str) -> Path:
            return tmp_path / job_id / relative_path

        def allocate_path(self, job_id: str, relative_path: str) -> str:
            return str(tmp_path / "output" / "converted.png")

    storage = MockStorage()

    with pytest.raises(FileNotFoundError):
        await task.run(job_id, params, storage)


@pytest.mark.asyncio
async def test_image_conversion_task_progress_callback(sample_image_path: Path, tmp_path: Path):
    """Test ImageConversionTask calls progress callback."""
    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(sample_image_path.read_bytes())

    params = ImageConversionParams(
        input_path=str(input_path),
        output_path="output/converted.png",
        format="png",
    )

    task = ImageConversionTask()
    job_id = "test-job-progress"

    class MockStorage:
        def resolve_path(self, job_id: str, relative_path: str) -> Path:
            return tmp_path / job_id / relative_path

        def allocate_path(self, job_id: str, relative_path: str) -> str:
            output_path = tmp_path / "output" / "converted.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return str(output_path)

    storage = MockStorage()

    progress_values = []

    def progress_callback(progress: int):
        progress_values.append(progress)

    await task.run(job_id, params, storage, progress_callback)

    assert 100 in progress_values


# ============================================================================
# ROUTE TESTS
# ============================================================================


def test_image_conversion_route_creation(api_client):
    """Test image_conversion route is registered."""
    response = api_client.get("/openapi.json")
    assert response.status_code == 200

    openapi = response.json()
    assert "/jobs/image_conversion" in openapi["paths"]


def test_image_conversion_route_job_submission(api_client, sample_image_path: Path):
    """Test job submission via image_conversion route."""
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/image_conversion",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"format": "png", "quality": 90, "priority": 5},
        )

    assert response.status_code == 200
    data = response.json()

    assert "job_id" in data
    assert data["status"] == "queued"
    assert data["task_type"] == "image_conversion"


def test_image_conversion_route_default_quality(api_client, sample_image_path: Path):
    """Test image_conversion route uses default quality."""
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/image_conversion",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"format": "png", "priority": 5},
        )

    assert response.status_code == 200


# ============================================================================
# INTEGRATION TEST (API → Worker)
# ============================================================================


@pytest.mark.integration
async def test_image_conversion_full_job_lifecycle(
    api_client, worker, job_repository, sample_image_path: Path, file_storage
):
    """Test complete flow: API → Repository → Worker → Output."""
    # 1. Submit job via API
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/image_conversion",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"format": "png", "quality": 85, "priority": 5},
        )

    assert response.status_code == 200
    job_id = response.json()["job_id"]

    # 2. Worker consumes job
    processed = await worker.run_once()
    assert processed == 1

    # 3. Verify completion
    job = job_repository.get(job_id)
    assert job is not None
    assert job.status == "completed"

    # 4. Validate output
    assert job.output is not None
    output = ImageConversionOutput.model_validate(job.output)
    assert isinstance(output, ImageConversionOutput)


@pytest.mark.integration
async def test_image_conversion_full_job_lifecycle_jpg_to_webp(
    api_client, worker, job_repository, sample_image_path: Path, file_storage
):
    """Test complete flow converting JPEG to WebP."""
    # 1. Submit job via API
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/image_conversion",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"format": "webp", "quality": 80, "priority": 6},
        )

    assert response.status_code == 200
    job_id = response.json()["job_id"]

    # 2. Worker consumes job
    processed = await worker.run_once()
    assert processed == 1

    # 3. Verify completion
    job = job_repository.get(job_id)
    assert job is not None
    assert job.status == "completed"
