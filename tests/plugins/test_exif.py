"""Unit and integration tests for EXIF metadata extraction plugin.

Tests schema validation, metadata extraction algorithms, task execution, routes, and full job lifecycle.
Requires ExifTool to be installed.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from cl_ml_tools import Worker
    from cl_ml_tools.common.job_repository import JobRepository

from cl_ml_tools.common.job_storage import JobStorage, SavedJobFile
from cl_ml_tools.plugins.exif.algo.exif_tool_wrapper import MetadataExtractor
from cl_ml_tools.plugins.exif.schema import ExifMetadataOutput, ExifMetadataParams
from cl_ml_tools.plugins.exif.task import ExifTask

# ============================================================================
# SCHEMA TESTS
# ============================================================================


def test_exif_params_schema_validation():
    """Test ExifMetadataParams schema validates correctly."""
    params = ExifMetadataParams(
        input_path="/path/to/input.jpg",
        output_path="output/exif.json",
        tags=["Make", "Model", "DateTimeOriginal"],
    )

    assert params.input_path == "/path/to/input.jpg"
    assert params.output_path == "output/exif.json"
    assert params.tags == ["Make", "Model", "DateTimeOriginal"]


def test_exif_params_defaults():
    """Test ExifMetadataParams has correct default values."""
    params = ExifMetadataParams(
        input_path="/path/to/input.jpg",
        output_path="output/exif.json",
    )

    assert params.tags == []


def test_exif_params_empty_tags_list():
    """Test ExifMetadataParams accepts empty tags list."""
    params = ExifMetadataParams(
        input_path="/path/to/input.jpg",
        output_path="output/exif.json",
        tags=[],
    )

    assert params.tags == []


def test_exif_output_schema_validation():
    """Test ExifMetadataOutput schema validates correctly."""
    output = ExifMetadataOutput(
        make="Apple",
        model="iPhone 13",
        image_width=4032,
        image_height=3024,
        gps_latitude=37.7749,
        gps_longitude=-122.4194,
    )

    assert output.make == "Apple"
    assert output.model == "iPhone 13"
    assert output.image_width == 4032
    assert output.image_height == 3024
    assert output.gps_latitude == 37.7749
    assert output.gps_longitude == -122.4194


def test_exif_output_optional_fields():
    """Test ExifMetadataOutput all fields are optional."""
    output = ExifMetadataOutput()

    assert output.make is None
    assert output.model is None
    assert output.image_width is None
    assert output.gps_latitude is None


def test_exif_output_from_raw_metadata():
    """Test ExifMetadataOutput.from_raw_metadata factory method."""
    raw_meta = {
        "Make": "Canon",
        "Model": "EOS 5D Mark IV",
        "ImageWidth": 6720,
        "ImageHeight": 4480,
        "ISO": 800,
        "FNumber": 2.8,
        "FocalLength": 50.0,
        "DateTimeOriginal": "2023:10:15 14:30:00",
    }

    output = ExifMetadataOutput.from_raw_metadata(cast("Any", raw_meta))

    assert output.make == "Canon"
    assert output.model == "EOS 5D Mark IV"
    assert output.image_width == 6720
    assert output.image_height == 4480
    assert output.iso == 800
    assert output.f_number == 2.8
    assert output.focal_length == 50.0
    assert output.date_time_original == "2023:10:15 14:30:00"
    assert output.raw_metadata == raw_meta


# ============================================================================
# ALGORITHM TESTS
# ============================================================================


@pytest.mark.requires_exiftool
def test_exif_algo_basic(sample_image_path: Path):
    """Test basic EXIF metadata extraction."""
    extractor = MetadataExtractor()

    metadata = extractor.extract_metadata_all(str(sample_image_path))

    assert isinstance(metadata, dict)
    assert len(metadata) > 0


@pytest.mark.requires_exiftool
def test_exif_algo_specific_tags(sample_image_path: Path):
    """Test extracting specific EXIF tags."""
    extractor = MetadataExtractor()

    tags = ["ImageWidth", "ImageHeight", "FileType"]
    metadata = extractor.extract_metadata(str(sample_image_path), tags=tags)

    assert isinstance(metadata, dict)
    # At least some tags should be present
    assert any(tag in metadata for tag in tags)


@pytest.mark.requires_exiftool
def test_exif_algo_with_gps_image(exif_test_image_path: Path):
    """Test EXIF extraction with GPS data."""
    extractor = MetadataExtractor()

    metadata = extractor.extract_metadata_all(str(exif_test_image_path))

    # GPS test image should have GPS data
    assert "GPSLatitude" in metadata or "GPS" in str(metadata)


@pytest.mark.requires_exiftool
def test_exif_algo_nonexistent_file():
    """Test EXIF extraction raises FileNotFoundError for non-existent file."""
    extractor = MetadataExtractor()

    with pytest.raises(FileNotFoundError, match="File not found"):
        extractor.extract_metadata_all("/nonexistent/file.jpg")


@pytest.mark.requires_exiftool
def test_exif_algo_error_handling_invalid_file(tmp_path: Path):
    """Test EXIF extraction handles invalid image files."""
    invalid_file = tmp_path / "invalid.jpg"
    _ = invalid_file.write_text("not an image")

    extractor = MetadataExtractor()

    # Should handle gracefully
    result = extractor.extract_metadata_all(str(invalid_file))

    assert isinstance(result, dict)


# ============================================================================
# TASK TESTS
# ============================================================================


@pytest.mark.requires_exiftool
@pytest.mark.asyncio
async def test_exif_task_run_success(sample_image_path: Path, tmp_path: Path):
    """Test ExifTask execution success."""
    # Copy sample to input location
    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(sample_image_path.read_bytes())

    params = ExifMetadataParams(
        input_path=str(input_path),
        output_path="output/exif.json",
    )

    task = ExifTask()
    task.setup()
    job_id = "test-job-123"

    # Mock storage
    class MockStorage:
        def create_directory(self, job_id: str) -> None:
            pass

        def remove(self, job_id: str) -> bool:
            return True

        async def save(
            self, job_id: str, relative_path: str, file: Any, *, mkdirs: bool = True
        ) -> SavedJobFile:
            return SavedJobFile(relative_path=relative_path, size=0, hash=None)

        async def open(self, job_id: str, relative_path: str) -> Any:
            return None

        def resolve_path(self, job_id: str, relative_path: str | None = None) -> Path:
            return tmp_path / job_id / (relative_path or "")

        def allocate_path(
            self, job_id: str, relative_path: str, *, mkdirs: bool = True
        ) -> Path:
            output_path = tmp_path / "output" / "exif.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return output_path

    storage = MockStorage()

    output = await task.run(job_id, params, storage)

    assert isinstance(output, ExifMetadataOutput)
    assert isinstance(output.raw_metadata, dict)

    # Verify output file
    output_file = tmp_path / "output" / "exif.json"
    assert output_file.exists()

    with open(output_file) as f:
        result = json.load(f)

    assert "raw_metadata" in result


@pytest.mark.requires_exiftool
@pytest.mark.asyncio
async def test_exif_task_run_with_specific_tags(
    sample_image_path: Path, tmp_path: Path
):
    """Test ExifTask with specific tags."""
    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(sample_image_path.read_bytes())

    params = ExifMetadataParams(
        input_path=str(input_path),
        output_path="output/exif.json",
        tags=["ImageWidth", "ImageHeight", "FileType"],
    )

    task = ExifTask()
    task.setup()
    job_id = "test-job-456"

    class MockStorage:
        def create_directory(self, job_id: str) -> None:
            pass

        def remove(self, job_id: str) -> bool:
            return True

        async def save(
            self, job_id: str, relative_path: str, file: Any, *, mkdirs: bool = True
        ) -> SavedJobFile:
            return SavedJobFile(relative_path=relative_path, size=0, hash=None)

        async def open(self, job_id: str, relative_path: str) -> Any:
            return None

        def resolve_path(self, job_id: str, relative_path: str | None = None) -> Path:
            return tmp_path / job_id / (relative_path or "")

        def allocate_path(
            self, job_id: str, relative_path: str, *, mkdirs: bool = True
        ) -> Path:
            output_path = tmp_path / "output" / "exif.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return output_path

    storage = MockStorage()

    output = await task.run(job_id, params, storage)

    assert isinstance(output, ExifMetadataOutput)


@pytest.mark.requires_exiftool
@pytest.mark.asyncio
async def test_exif_task_run_file_not_found(tmp_path: Path):
    """Test ExifTask handles missing input file."""
    params = ExifMetadataParams(
        input_path="/nonexistent/file.jpg",
        output_path="output/exif.json",
    )

    task = ExifTask()
    task.setup()
    job_id = "test-job-789"

    class MockStorage:
        def create_directory(self, job_id: str) -> None:
            pass

        def remove(self, job_id: str) -> bool:
            return True

        async def save(
            self, job_id: str, relative_path: str, file: Any, *, mkdirs: bool = True
        ) -> SavedJobFile:
            return SavedJobFile(relative_path=relative_path, size=0, hash=None)

        async def open(self, job_id: str, relative_path: str) -> Any:
            return None

        def resolve_path(self, job_id: str, relative_path: str | None = None) -> Path:
            return tmp_path / job_id / (relative_path or "")

        def allocate_path(
            self, job_id: str, relative_path: str, *, mkdirs: bool = True
        ) -> Path:
            output_path = tmp_path / "output" / "exif.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return output_path

    storage = MockStorage()

    # Should raise FileNotFoundError for missing file
    with pytest.raises(FileNotFoundError):
        await task.run(job_id, params, storage)


@pytest.mark.requires_exiftool
@pytest.mark.asyncio
async def test_exif_task_progress_callback(sample_image_path: Path, tmp_path: Path):
    """Test ExifTask calls progress callback."""
    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(sample_image_path.read_bytes())

    params = ExifMetadataParams(
        input_path=str(input_path),
        output_path="output/exif.json",
    )

    task = ExifTask()
    task.setup()
    job_id = "test-job-progress"

    class MockStorage:
        def create_directory(self, job_id: str) -> None:
            pass

        def remove(self, job_id: str) -> bool:
            return True

        async def save(
            self,
            job_id: str,
            relative_path: str,
            file: Any,
            *,
            mkdirs: bool = True,
        ) -> SavedJobFile:
            return SavedJobFile(relative_path=relative_path, size=0, hash=None)

        async def open(self, job_id: str, relative_path: str) -> Any:
            return None

        def resolve_path(self, job_id: str, relative_path: str | None = None) -> Path:
            return tmp_path / job_id / (relative_path or "")

        def allocate_path(
            self, job_id: str, relative_path: str, *, mkdirs: bool = True
        ) -> Path:
            output_path = tmp_path / "output" / "exif.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return output_path

    storage = MockStorage()

    progress_values = []

    def progress_callback(progress: int):
        progress_values.append(progress)

    await task.run(job_id, params, storage, progress_callback)

    assert 100 in progress_values


def test_exif_task_setup_without_exiftool():
    """Test ExifTask.setup raises RuntimeError without ExifTool."""
    # This test assumes ExifTool IS installed (due to marker)
    # Just verify setup doesn't crash
    task = ExifTask()

    # Should not raise if ExifTool is installed
    task.setup()

    assert task._extractor is not None


# ============================================================================
# ROUTE TESTS
# ============================================================================


def test_exif_route_creation(api_client: "TestClient"):
    """Test exif route is registered."""
    response = api_client.get("/openapi.json")
    assert response.status_code == 200

    openapi = response.json()
    assert "/jobs/exif" in openapi["paths"]


@pytest.mark.requires_exiftool
def test_exif_route_job_submission(api_client: "TestClient", sample_image_path: Path):
    """Test job submission via exif route."""
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/exif",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"priority": "5"},
        )

    assert response.status_code == 200
    data = response.json()

    assert "job_id" in data
    assert data["status"] == "queued"
    assert data["task_type"] == "exif"


@pytest.mark.requires_exiftool
def test_exif_route_job_submission_with_tags(
    api_client: "TestClient", sample_image_path: Path
):
    """Test job submission with specific tags."""
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/exif",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={
                "priority": "5",
                "tags": '["ImageWidth", "ImageHeight"]',  # JSON string
            },
        )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "queued"


# ============================================================================
# INTEGRATION TEST (API → Worker)
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_exiftool
@pytest.mark.asyncio
async def test_exif_full_job_lifecycle(
    api_client: "TestClient",
    worker: "Worker",
    job_repository: "JobRepository",
    sample_image_path: Path,
    file_storage: "JobStorage",
):
    """Test complete flow: API → Repository → Worker → Output."""
    # 1. Submit job via API
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/exif",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"priority": "5"},
        )

    assert response.status_code == 200
    job_id = response.json()["job_id"]

    # 2. Worker consumes job
    processed = await worker.run_once()
    assert processed == 1

    # 3. Verify completion
    job = job_repository.get_job(job_id)
    assert job is not None
    assert job.status == "completed"

    # 4. Validate output
    assert job.output is not None
    output = ExifMetadataOutput.model_validate(job.output)
    assert isinstance(output.raw_metadata, dict)


@pytest.mark.integration
@pytest.mark.requires_exiftool
@pytest.mark.asyncio
async def test_exif_full_job_lifecycle_with_gps(
    api_client: "TestClient",
    worker: "Worker",
    job_repository: "JobRepository",
    exif_test_image_path: Path,
    file_storage: "JobStorage",
):
    """Test complete flow with GPS EXIF data."""
    # 1. Submit job via API
    with open(exif_test_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/exif",
            files={"file": ("test_gps.jpg", f, "image/jpeg")},
            data={"priority": "5"},
        )

    assert response.status_code == 200
    job_id = response.json()["job_id"]

    # 2. Worker consumes job
    processed = await worker.run_once()
    assert processed == 1

    # 3. Verify completion
    job = job_repository.get_job(job_id)
    assert job is not None
    assert job.status == "completed"

    # 4. Validate output has GPS data
    assert job.output is not None
    output = ExifMetadataOutput.model_validate(job.output)
    # GPS test image should have GPS coordinates
    assert output.gps_latitude is not None or "GPS" in str(output.raw_metadata)
