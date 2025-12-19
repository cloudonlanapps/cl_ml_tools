"""Test configuration and fixtures for cl_ml_tools.

This module provides:
- Pytest configuration (markers, dependency checks)
- Session-scoped fixtures (test media validation, model downloads)
- Function-scoped fixtures (temp dirs, sample files, mock services)
- Integration test fixtures (API client, worker, repositories)
"""

import hashlib
import shutil
from pathlib import Path
from typing import Any, cast, override

import pytest
from fastapi.testclient import TestClient

# Test directory paths
TESTS_DIR = Path(__file__).parent
TEST_MEDIA_DIR = TESTS_DIR / "test_media"
MANIFEST_FILE = TESTS_DIR / "MANIFEST.md5"


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_addoption(parser):
    """Add custom command line options and ini values."""
    parser.addini(
        "test_storage_base_dir",
        help="Base directory for test storage",
        default="/tmp/cl_ml_tools_test_storage",
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_ffmpeg: requires FFmpeg to be installed",
    )
    config.addinivalue_line(
        "markers",
        "requires_exiftool: requires ExifTool to be installed",
    )
    config.addinivalue_line(
        "markers",
        "requires_models: requires ML models downloaded",
    )
    config.addinivalue_line(
        "markers",
        "integration: full integration tests (API → Worker)",
    )


def pytest_runtest_setup(item):
    """Check dependencies before running tests - FAIL if missing (not skip)."""
    # Check FFmpeg
    if item.get_closest_marker("requires_ffmpeg") and not shutil.which("ffmpeg"):
        pytest.fail(
            "FFmpeg not installed. "
            "Install: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)\n"
            "Or exclude with: pytest -m 'not requires_ffmpeg'",
            pytrace=False,
        )

    # Check ExifTool
    if item.get_closest_marker("requires_exiftool") and not shutil.which("exiftool"):
        pytest.fail(
            "ExifTool not installed. "
            "Install: brew install exiftool (macOS) or "
            "apt-get install libimage-exiftool-perl (Linux)\n"
            "Or exclude with: pytest -m 'not requires_exiftool'",
            pytrace=False,
        )

    # Check ML models (verify models directory exists with expected files)
    if item.get_closest_marker("requires_models"):
        from pathlib import Path

        model_dir = Path.home() / ".cache" / "cl_ml_tools" / "models"
        if not model_dir.exists() or not any(model_dir.iterdir()):
            pytest.fail(
                "ML models not downloaded. Models are downloaded on first use.\n"
                "Or exclude with: pytest -m 'not requires_models'",
                pytrace=False,
            )


# ============================================================================
# Session-Scoped Fixtures (Run Once)
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def validate_test_media():
    """
    Validate test media manifest before any tests run.

    This fixture runs automatically for all test sessions and ensures
    that test_media directory exists and matches MANIFEST.md5.
    """
    if not TEST_MEDIA_DIR.exists():
        pytest.exit(
            f"Test media directory not found: {TEST_MEDIA_DIR}\n"
            f"Run: python tests/setup_test_media.py",
            returncode=1,
        )

    if not MANIFEST_FILE.exists():
        pytest.exit(
            f"Test media manifest not found: {MANIFEST_FILE}\n"
            f"Run: python tests/setup_test_media.py",
            returncode=1,
        )

    # Validate checksums
    errors = []
    with open(MANIFEST_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue

            expected_md5, relative_path = parts
            file_path = TESTS_DIR / relative_path

            if not file_path.exists():
                errors.append(f"Missing: {relative_path}")
                continue

            # Calculate actual MD5
            md5_hash = hashlib.md5()
            with open(file_path, "rb") as f2:
                for chunk in iter(lambda: f2.read(4096), b""):
                    md5_hash.update(chunk)
            actual_md5 = md5_hash.hexdigest()

            if actual_md5 != expected_md5:
                errors.append(
                    f"Checksum mismatch: {relative_path}\n"
                    f"  Expected: {expected_md5}\n"
                    f"  Actual:   {actual_md5}",
                )

    if errors:
        pytest.exit(
            "Test media validation failed:\n" + "\n".join(errors) + "\n\n"
            "Run: python tests/setup_test_media.py",
            returncode=1,
        )


# ============================================================================
# Function-Scoped Fixtures (Run Per Test)
# ============================================================================


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Provide clean temporary directory for test outputs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_image_path() -> Path:
    """Provide path to a real test image.

    Raises:
        pytest.fail: If no test images available (ensures CI/CD catches missing setup)
    """
    # Use first available image from test_media
    images_dir = TEST_MEDIA_DIR / "images"
    if images_dir.exists():
        images = list(images_dir.glob("*.jpg"))
        if images:
            return images[0]

    pytest.fail("No test images available.\nRun: python tests/setup_test_media.py")


@pytest.fixture
def sample_video_path() -> Path:
    """Provide path to a test video, generating one if needed.

    For HLS streaming tests, generates 30-second HD videos (1280x720)
    to ensure multiple .ts segments are created.

    Raises:
        pytest.fail: If video cannot be generated (ensures CI/CD catches issues)
    """
    from tests.utils.video_generator import ensure_hls_test_videos_exist

    videos_dir = TEST_MEDIA_DIR / "videos"

    try:
        videos = ensure_hls_test_videos_exist(videos_dir, count=1)
        if videos:
            return videos[0]
    except ImportError as exc:
        pytest.fail(
            f"OpenCV (cv2) not installed. Required for video generation.\n"
            f"Install with: pip install opencv-python\n"
            f"Error: {exc}",
        )
    except RuntimeError as exc:
        pytest.fail(f"Video generation failed: {exc}")

    pytest.fail("No test videos available and generation failed")


@pytest.fixture
def exif_test_image_path() -> Path:
    """Provide path to image with known EXIF metadata.

    Raises:
        pytest.fail: If EXIF test images not generated (ensures CI/CD catches missing setup)
    """
    exif_dir = TEST_MEDIA_DIR / "exif_generated"
    if exif_dir.exists():
        images = list(exif_dir.glob("with_gps.jpg"))
        if images:
            return images[0]

    pytest.fail("EXIF test images not generated.\nRun: python tests/generate_exif_test_media.py")


@pytest.fixture
def synthetic_image(tmp_path: Path) -> Path:
    """Generate synthetic test image using PIL."""
    from PIL import Image, ImageDraw

    output_path = tmp_path / "synthetic.jpg"

    # Create simple test image (800x600 with gradient)
    img = Image.new("RGB", (800, 600), color=(73, 109, 137))
    draw = ImageDraw.Draw(img)

    # Add some patterns to make it interesting
    for i in range(0, 800, 50):
        draw.line([(i, 0), (i, 600)], fill=(255, 255, 255), width=2)
    for i in range(0, 600, 50):
        draw.line([(0, i), (800, i)], fill=(255, 255, 255), width=2)

    # Draw a circle in center
    draw.ellipse([300, 200, 500, 400], fill=(200, 100, 100))

    img.save(output_path, "JPEG", quality=85)

    return output_path


# ============================================================================
# Mock Service Fixtures
# ============================================================================


@pytest.fixture
def job_repository():
    """Provide in-memory job repository for testing."""
    from collections.abc import Sequence

    from cl_ml_tools.common.job_repository import JobRepository
    from cl_ml_tools.common.schema_job_record import JobRecord, JobRecordUpdate, JobStatus

    class InMemoryJobRepository(JobRepository):
        """In-memory implementation for testing."""

        def __init__(self):
            self._jobs: dict[str, JobRecord] = {}

        def create(self, job: JobRecord) -> JobRecord:
            self._jobs[job.job_id] = job
            return job

        def get(self, job_id: str) -> JobRecord | None:
            return self._jobs.get(job_id)

        def update(self, job_id: str, **kwargs) -> JobRecord | None:
            if job_id not in self._jobs:
                return None
            job = self._jobs[job_id]
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            return job

        def list_pending(self, limit: int = 100) -> list[JobRecord]:
            return [job for job in self._jobs.values() if job.status == JobStatus.queued][:limit]

        def delete(self, job_id: str) -> bool:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
            return False

        # Protocol-required methods
        @override
        def add_job(
            self,
            job: JobRecord,
            created_by: str | None = None,
            priority: int | None = None,
        ) -> bool:
            """Protocol alias for save."""
            self._jobs[job.job_id] = job
            return True

        @override
        def get_job(self, job_id: str) -> JobRecord | None:
            """Protocol alias for get."""
            return self._jobs.get(job_id)

        @override
        def update_job(self, job_id: str, updates: JobRecordUpdate) -> bool:
            """Update job fields (Protocol-compliant)."""
            if job_id not in self._jobs:
                return False
            job = self._jobs[job_id]
            # Convert Pydantic model to dict if needed
            updates_dict: dict[str, Any] = (
                updates.model_dump(exclude_none=True)
                if hasattr(updates, "model_dump")
                else cast("dict[str, Any]", cast("Any", updates))
            )
            for key, value in updates_dict.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            return True

        @override
        def fetch_next_job(self, task_types: Sequence[str]) -> JobRecord | None:
            """Atomically fetch and claim next queued job."""
            for job in self._jobs.values():
                if job.status == JobStatus.queued and job.task_type in task_types:
                    job.status = JobStatus.processing
                    return job
            return None

        @override
        def delete_job(self, job_id: str) -> bool:
            """Delete job (Protocol alias for delete)."""
            return self.delete(job_id)

    return InMemoryJobRepository()


@pytest.fixture
def file_storage(tmp_path: Path, pytestconfig):
    """Provide file storage for testing.

    Configuration priority:
    1. TEST_STORAGE_DIR environment variable
    2. pytest.ini test_storage_base_dir option
    3. Default: tmp_path / "file_storage"
    """
    import os

    from cl_ml_tools.common.file_storage_impl import LocalFileStorage

    # Check environment variable first
    env_storage = os.environ.get("TEST_STORAGE_DIR")
    if env_storage:
        storage_dir = Path(env_storage)
        storage_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Check pytest.ini configuration
        ini_storage = pytestconfig.getini("test_storage_base_dir")
        if ini_storage:
            storage_dir = Path(ini_storage)
            storage_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Default to tmp_path
            storage_dir = tmp_path / "file_storage"
            storage_dir.mkdir()

    return LocalFileStorage(base_dir=storage_dir)


@pytest.fixture
def worker(job_repository, file_storage):
    """Provide Worker instance for integration tests."""
    from cl_ml_tools import Worker

    return Worker(
        repository=job_repository,
        job_storage=file_storage,
    )


@pytest.fixture
def api_client(job_repository, file_storage):
    """Provide FastAPI TestClient for route testing."""
    from fastapi import FastAPI

    from cl_ml_tools import create_master_router

    app = FastAPI()

    # Create router with test dependencies
    def get_current_user():
        return None

    router = create_master_router(
        repository=job_repository,
        file_storage=file_storage,
        get_current_user=get_current_user,
    )

    app.include_router(router)

    return TestClient(app)


# ============================================================================
# Helper Fixtures
# ============================================================================


@pytest.fixture
def sample_job_params():
    """Provide sample job parameters for testing."""
    from cl_ml_tools.common.schema_job import BaseJobParams

    class TestJobParams(BaseJobParams):
        input_path: str
        output_path: str

    return TestJobParams(
        input_path="/test/input.jpg",
        output_path="output/test.json",
    )
