"""Unit and integration tests for media thumbnail plugin.

Tests schema validation, image & video thumbnail generation, aspect ratio handling, task execution, routes, and full job lifecycle.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from PIL import Image

from cl_ml_tools.common.file_storage import SavedJobFile

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from cl_ml_tools import Worker
    from cl_ml_tools.common.file_storage import JobStorage
    from cl_ml_tools.common.job_repository import JobRepository

from cl_ml_tools.plugins.media_thumbnail.algo.image_thumbnail import (
    image_thumbnail,
)
from cl_ml_tools.plugins.media_thumbnail.algo.video_thumbnail import (
    video_thumbnail,
)
from cl_ml_tools.plugins.media_thumbnail.schema import (
    MediaThumbnailOutput,
    MediaThumbnailParams,
)
from cl_ml_tools.plugins.media_thumbnail.task import MediaThumbnailTask

# ============================================================================
# SCHEMA TESTS
# ============================================================================


def test_media_thumbnail_params_schema_validation():
    """Test MediaThumbnailParams schema validates correctly."""
    params = MediaThumbnailParams(
        input_path="/path/to/input.jpg",
        output_path="output/thumbnail.jpg",
        width=200,
        height=150,
        maintain_aspect_ratio=True,
    )

    assert params.input_path == "/path/to/input.jpg"
    assert params.output_path == "output/thumbnail.jpg"
    assert params.width == 200
    assert params.height == 150
    assert params.maintain_aspect_ratio is True


def test_media_thumbnail_params_defaults():
    """Test MediaThumbnailParams has correct default values."""
    params = MediaThumbnailParams(
        input_path="/path/to/input.jpg",
        output_path="output/thumbnail.jpg",
    )

    assert params.width is None
    assert params.height is None
    assert params.maintain_aspect_ratio is True


def test_media_thumbnail_params_width_only():
    """Test MediaThumbnailParams with width only."""
    params = MediaThumbnailParams(
        input_path="/path/to/input.jpg",
        output_path="output/thumbnail.jpg",
        width=300,
    )

    assert params.width == 300
    assert params.height is None


def test_media_thumbnail_output_schema_validation():
    """Test MediaThumbnailOutput schema validates correctly."""
    output = MediaThumbnailOutput(media_type="image")

    assert output.media_type == "image"


# ============================================================================
# ALGORITHM TESTS - Image Thumbnails
# ============================================================================


def test_image_thumbnail_algo_basic(sample_image_path: Path, tmp_path: Path):
    """Test basic image thumbnail creation."""
    output_path = tmp_path / "thumbnail.jpg"

    image_thumbnail(
        input_path=str(sample_image_path),
        output_path=str(output_path),
        width=200,
        height=150,
    )

    assert output_path.exists()

    # Verify dimensions
    with Image.open(output_path) as img:
        assert img.width <= 200
        assert img.height <= 150


def test_image_thumbnail_algo_width_only(sample_image_path: Path, tmp_path: Path):
    """Test thumbnail creation with width only (maintains aspect ratio)."""
    output_path = tmp_path / "thumbnail.jpg"

    image_thumbnail(
        input_path=str(sample_image_path),
        output_path=str(output_path),
        width=300,
        height=None,
    )

    assert output_path.exists()

    with Image.open(output_path) as img:
        assert img.width == 300


def test_image_thumbnail_algo_height_only(sample_image_path: Path, tmp_path: Path):
    """Test thumbnail creation with height only (maintains aspect ratio)."""
    output_path = tmp_path / "thumbnail.jpg"

    image_thumbnail(
        input_path=str(sample_image_path),
        output_path=str(output_path),
        width=None,
        height=200,
    )

    assert output_path.exists()

    with Image.open(output_path) as img:
        assert img.height == 200


def test_image_thumbnail_algo_aspect_ratio_maintained(sample_image_path: Path, tmp_path: Path):
    """Test thumbnail maintains aspect ratio."""
    # Get original aspect ratio
    with Image.open(sample_image_path) as img:
        original_ratio = img.width / img.height

    output_path = tmp_path / "thumbnail.jpg"

    image_thumbnail(
        input_path=str(sample_image_path),
        output_path=str(output_path),
        width=200,
        height=200,
    )

    with Image.open(output_path) as img:
        thumbnail_ratio = img.width / img.height
        # Ratios should be close (within 1%)
        assert abs(thumbnail_ratio - original_ratio) / original_ratio < 0.01


def test_image_thumbnail_algo_larger_than_original(sample_image_path: Path, tmp_path: Path):
    """Test thumbnail creation when requested size is larger than original."""
    output_path = tmp_path / "thumbnail.jpg"

    image_thumbnail(
        input_path=str(sample_image_path),
        output_path=str(output_path),
        width=10000,  # Much larger than any test image
        height=None,
    )

    assert output_path.exists()

    # Thumbnail should not exceed original size
    with Image.open(sample_image_path) as orig, Image.open(output_path) as thumb:
        assert thumb.width <= orig.width


def test_image_thumbnail_algo_file_not_found(tmp_path: Path):
    """Test thumbnail creation with non-existent file."""
    output_path = tmp_path / "thumbnail.jpg"

    with pytest.raises(FileNotFoundError):
        image_thumbnail(
            input_path="/nonexistent/file.jpg",
            output_path=str(output_path),
            width=200,
            height=None,
        )


# ============================================================================
# ALGORITHM TESTS - Video Thumbnails
# ============================================================================


@pytest.mark.requires_ffmpeg
def test_video_thumbnail_algo_basic(sample_video_path: Path, tmp_path: Path):
    """Test basic video thumbnail creation."""
    output_path = tmp_path / "video_thumbnail.jpg"

    video_thumbnail(
        input_path=str(sample_video_path),
        output_path=str(output_path),
        width=256,
        height=256,
    )

    assert output_path.exists()
    with Image.open(output_path) as img:
        assert img.width <= 256
        assert img.height <= 256


@pytest.mark.requires_ffmpeg
def test_video_thumbnail_algo_width_only(sample_video_path: Path, tmp_path: Path):
    """Test video thumbnail with width only."""
    output_path = tmp_path / "video_thumbnail_w.jpg"

    video_thumbnail(
        input_path=str(sample_video_path),
        output_path=str(output_path),
        width=300,
    )

    assert output_path.exists()
    with Image.open(output_path) as img:
        assert img.width == 300


@pytest.mark.requires_ffmpeg
def test_video_thumbnail_algo_height_only(sample_video_path: Path, tmp_path: Path):
    """Test video thumbnail with height only."""
    output_path = tmp_path / "video_thumbnail_h.jpg"

    video_thumbnail(
        input_path=str(sample_video_path),
        output_path=str(output_path),
        height=200,
    )

    assert output_path.exists()
    with Image.open(output_path) as img:
        assert img.height == 200


@pytest.mark.requires_ffmpeg
def test_video_thumbnail_algo_output_dir_missing(sample_video_path: Path, tmp_path: Path):
    """Test video thumbnail raises FileNotFoundError if output directory missing."""
    output_path = tmp_path / "nonexistent_dir" / "thumb.jpg"

    with pytest.raises(FileNotFoundError, match="Output directory not found"):
        video_thumbnail(
            input_path=str(sample_video_path),
            output_path=str(output_path),
        )


@pytest.mark.requires_ffmpeg
def test_video_thumbnail_algo_input_missing(tmp_path: Path):
    """Test video thumbnail raises FileNotFoundError if input missing."""
    input_path = tmp_path / "missing.mp4"
    output_path = tmp_path / "thumb.jpg"

    with pytest.raises(FileNotFoundError, match="Input file not found"):
        video_thumbnail(
            input_path=str(input_path),
            output_path=str(output_path),
        )


# ============================================================================
# TASK TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_media_thumbnail_task_run_success_image(sample_image_path: Path, tmp_path: Path):
    """Test MediaThumbnailTask execution with image."""
    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(sample_image_path.read_bytes())

    params = MediaThumbnailParams(
        input_path=str(input_path),
        output_path="output/thumbnail.jpg",
        width=200,
        height=150,
    )

    task = MediaThumbnailTask()
    job_id = "test-job-123"

    class MockStorage:
        def create_directory(self, _id: str) -> None:
            pass

        def remove(self, _id: str) -> bool:
            return True

        async def save(self, _id, _path, _file, **_k) -> SavedJobFile:
            return SavedJobFile(relative_path=_path, size=0)

        async def open(self, _id, _path) -> Any:
            return None

        def resolve_path(self, job_id: str, relative_path: str | None = None) -> Path:
            return tmp_path / job_id / (relative_path or "")

        def allocate_path(self, job_id: str, relative_path: str) -> Path:
            output_path = tmp_path / "output" / "thumbnail.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return output_path

    storage = MockStorage()

    output = await task.run(job_id, params, storage)

    assert isinstance(output, MediaThumbnailOutput)
    assert output.media_type == "image"

    # Verify output file
    output_file = tmp_path / "output" / "thumbnail.jpg"
    assert output_file.exists()


@pytest.mark.asyncio
async def test_media_thumbnail_task_run_file_not_found(tmp_path: Path):
    """Test MediaThumbnailTask raises FileNotFoundError for missing input."""
    params = MediaThumbnailParams(
        input_path="/nonexistent/file.jpg",
        output_path="output/thumbnail.jpg",
        width=200,
    )

    task = MediaThumbnailTask()
    job_id = "test-job-789"

    class MockStorage:
        def create_directory(self, job_id: str) -> None:
            pass

        def remove(self, job_id: str) -> bool:
            return True

        async def save(
            self, job_id: str, relative_path: str, file: Any, *, mkdirs: bool = True
        ) -> SavedJobFile:
            return SavedJobFile(relative_path=relative_path, size=0)

        async def open(self, job_id: str, relative_path: str) -> Any:
            return None

        def resolve_path(self, job_id: str, relative_path: str | None = None) -> Path:
            return tmp_path / job_id / (relative_path or "")

        def allocate_path(self, job_id: str, relative_path: str, *, mkdirs: bool = True) -> Path:
            return tmp_path / "output" / "thumbnail.jpg"

    storage = MockStorage()

    with pytest.raises(FileNotFoundError):
        await task.run(job_id, params, storage)


@pytest.mark.asyncio
async def test_media_thumbnail_task_progress_callback(sample_image_path: Path, tmp_path: Path):
    """Test MediaThumbnailTask calls progress callback."""
    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(sample_image_path.read_bytes())

    params = MediaThumbnailParams(
        input_path=str(input_path),
        output_path="output/thumbnail.jpg",
        width=200,
    )

    task = MediaThumbnailTask()
    job_id = "test-job-progress"

    class MockStorage:
        def create_directory(self, _id: str) -> None:
            pass

        def remove(self, _id: str) -> bool:
            return True

        async def save(self, _id, _path, _file, **_k) -> SavedJobFile:
            return SavedJobFile(relative_path=_path, size=0)

        async def open(self, _id, _path) -> Any:
            return None

        def resolve_path(self, job_id: str, relative_path: str | None = None) -> Path:
            return tmp_path / job_id / (relative_path or "")

        def allocate_path(self, job_id: str, relative_path: str) -> Path:
            output_path = tmp_path / "output" / "thumbnail.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return output_path

    storage = MockStorage()

    progress_values = []

    def progress_callback(progress: int):
        progress_values.append(progress)

    await task.run(job_id, params, storage, progress_callback)

    assert 100 in progress_values


# ============================================================================
# ROUTE TESTS
# ============================================================================


def test_media_thumbnail_route_creation(api_client: "TestClient"):
    """Test media_thumbnail route is registered."""
    response = api_client.get("/openapi.json")
    assert response.status_code == 200

    openapi = response.json()
    assert "/jobs/media_thumbnail" in openapi["paths"]


def test_media_thumbnail_route_job_submission(api_client: "TestClient", sample_image_path: Path):
    """Test job submission via media_thumbnail route."""
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/media_thumbnail",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"width": "200", "height": "150", "priority": "5"},
        )

    assert response.status_code == 200
    data = response.json()

    assert "job_id" in data
    assert data["status"] == "queued"
    assert data["task_type"] == "media_thumbnail"


def test_media_thumbnail_route_width_only(api_client: "TestClient", sample_image_path: Path):
    """Test job submission with width only."""
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/media_thumbnail",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"width": "300", "priority": "5"},
        )

    assert response.status_code == 200


# ============================================================================
# INTEGRATION TEST (API → Worker)
# ============================================================================


@pytest.mark.integration
async def test_media_thumbnail_full_job_lifecycle(
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
            "/jobs/media_thumbnail",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"width": "200", "height": "150", "priority": "5"},
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
    output = MediaThumbnailOutput.model_validate(job.output)
    assert output.media_type == "image"


@pytest.mark.integration
async def test_media_thumbnail_full_job_lifecycle_aspect_ratio(
    api_client: "TestClient",
    worker: "Worker",
    job_repository: "JobRepository",
    sample_image_path: Path,
    file_storage: "JobStorage",
):
    """Test complete flow with aspect ratio preservation."""
    # 1. Submit job via API
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/media_thumbnail",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"width": "300", "maintain_aspect_ratio": "true", "priority": "6"},
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
