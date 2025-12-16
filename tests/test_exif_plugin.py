"""Comprehensive test suite for EXIF metadata extraction plugin."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from cl_ml_tools.common.schemas import Job
from cl_ml_tools.plugins.exif.algo.exif_tool_wrapper import MetadataExtractor
from cl_ml_tools.plugins.exif.schema import ExifMetadata, ExifParams
from cl_ml_tools.plugins.exif.task import ExifTask

# ============================================================================
# Test Fixtures
# ============================================================================


class MockProgressCallback:
    """Mock progress callback for testing."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, progress: int) -> None:
        self.calls.append(progress)


@pytest.fixture
def sample_image_with_exif(tmp_path: Path) -> str:
    """Create a test image with EXIF metadata using PIL."""
    image_path = tmp_path / "test_image_with_exif.jpg"

    # Create a simple image
    img = Image.new("RGB", (800, 600), color=(73, 109, 137))

    # Create EXIF data
    from PIL import Image as PILImage

    exif = PILImage.Exif()

    # Add some common EXIF tags (using tag codes)
    exif[0x010F] = "TestCamera"  # Make
    exif[0x0110] = "TestModel X1"  # Model
    exif[0x0112] = 1  # Orientation (normal)
    exif[0x9003] = "2024:01:15 10:30:00"  # DateTimeOriginal

    # Save image with EXIF
    img.save(str(image_path), "JPEG", exif=exif)

    return str(image_path)


@pytest.fixture
def sample_image_without_exif(tmp_path: Path) -> str:
    """Create a test image without EXIF metadata."""
    image_path = tmp_path / "test_image_no_exif.jpg"

    # Create and save image without EXIF
    img = Image.new("RGB", (640, 480), color=(200, 100, 50))
    img.save(str(image_path), "JPEG")

    return str(image_path)


@pytest.fixture
def exif_task() -> ExifTask:
    """Create ExifTask instance."""
    return ExifTask()


@pytest.fixture
def mock_progress_callback() -> MockProgressCallback:
    """Create mock progress callback."""
    return MockProgressCallback()


# ============================================================================
# Test Class 1: Schema Validation
# ============================================================================


class TestExifParams:
    """Test ExifParams schema validation."""

    def test_default_params(self) -> None:
        """Test default ExifParams values."""
        params = ExifParams(input_paths=["/test/file.jpg"], output_paths=[])

        assert params.input_paths == ["/test/file.jpg"]
        assert params.output_paths == []
        assert params.tags == []

    def test_custom_tags(self) -> None:
        """Test ExifParams with custom tags."""
        params = ExifParams(
            input_paths=["/test/file.jpg"],
            output_paths=[],
            tags=["Make", "Model", "DateTimeOriginal"],
        )

        assert params.tags == ["Make", "Model", "DateTimeOriginal"]

    def test_empty_tags_list(self) -> None:
        """Test ExifParams with empty tags list (extracts all)."""
        params = ExifParams(input_paths=["/test/file.jpg"], output_paths=[], tags=[])

        assert params.tags == []


class TestExifMetadata:
    """Test ExifMetadata schema and conversion."""

    def test_empty_metadata(self) -> None:
        """Test ExifMetadata with no data."""
        metadata = ExifMetadata()

        assert metadata.make is None
        assert metadata.model is None
        assert metadata.date_time_original is None
        assert metadata.raw_metadata == {}

    def test_from_raw_metadata(self) -> None:
        """Test conversion from raw metadata dict."""
        raw_meta = {
            "Make": "Canon",
            "Model": "EOS R5",
            "DateTimeOriginal": "2024:01:15 10:30:00",
            "ImageWidth": 6000,
            "ImageHeight": 4000,
            "ISO": 400,
            "FNumber": 2.8,
            "ExposureTime": "1/125",
            "FocalLength": 50.0,
            "GPSLatitude": 37.7749,
            "GPSLongitude": -122.4194,
        }

        metadata = ExifMetadata.from_raw_metadata(raw_meta)

        assert metadata.make == "Canon"
        assert metadata.model == "EOS R5"
        assert metadata.date_time_original == "2024:01:15 10:30:00"
        assert metadata.image_width == 6000
        assert metadata.image_height == 4000
        assert metadata.iso == 400
        assert metadata.f_number == 2.8
        assert metadata.exposure_time == "1/125"
        assert metadata.focal_length == 50.0
        assert metadata.gps_latitude == 37.7749
        assert metadata.gps_longitude == -122.4194
        assert metadata.raw_metadata == raw_meta

    def test_partial_metadata(self) -> None:
        """Test conversion with partial metadata."""
        raw_meta = {
            "Make": "Nikon",
            "ISO": 800,
        }

        metadata = ExifMetadata.from_raw_metadata(raw_meta)

        assert metadata.make == "Nikon"
        assert metadata.model is None  # Not in raw_meta
        assert metadata.iso == 800
        assert metadata.raw_metadata == raw_meta


# ============================================================================
# Test Class 2: MetadataExtractor Algorithm
# ============================================================================


class TestMetadataExtractor:
    """Test MetadataExtractor algorithm (ExifTool wrapper)."""

    def test_exiftool_availability_check(self) -> None:
        """Test ExifTool availability check."""
        extractor = MetadataExtractor()

        # If this doesn't raise, ExifTool is available
        assert extractor.is_exiftool_available() is True

    def test_exiftool_not_available(self) -> None:
        """Test error when ExifTool is not available."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="ExifTool is not installed"):
                MetadataExtractor()

    def test_extract_specific_tags(
        self, sample_image_with_exif: str, tmp_path: Path
    ) -> None:
        """Test extracting specific EXIF tags."""
        # Skip if exiftool not installed
        try:
            subprocess.run(
                ["exiftool", "-ver"], check=True, capture_output=True, timeout=5
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("ExifTool not installed")

        extractor = MetadataExtractor()

        tags = ["Make", "Model", "Orientation"]
        metadata = extractor.extract_metadata(sample_image_with_exif, tags=tags)

        # Should have some metadata
        assert isinstance(metadata, dict)
        # May have Make/Model if exiftool can read PIL-generated EXIF
        # (this is platform-dependent)

    def test_extract_all_metadata(
        self, sample_image_with_exif: str, tmp_path: Path
    ) -> None:
        """Test extracting all EXIF tags."""
        # Skip if exiftool not installed
        try:
            subprocess.run(
                ["exiftool", "-ver"], check=True, capture_output=True, timeout=5
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("ExifTool not installed")

        extractor = MetadataExtractor()

        metadata = extractor.extract_metadata_all(sample_image_with_exif)

        # Should have some metadata (at least file info)
        assert isinstance(metadata, dict)
        assert len(metadata) > 0

    def test_extract_from_nonexistent_file(self) -> None:
        """Test extraction from non-existent file."""
        extractor = MetadataExtractor()

        with pytest.raises(FileNotFoundError):
            extractor.extract_metadata("/nonexistent/file.jpg", tags=["Make"])

    def test_extract_no_tags_provided(self, sample_image_with_exif: str) -> None:
        """Test extraction with empty tags list."""
        extractor = MetadataExtractor()

        # Empty tags should return empty dict (per current implementation)
        metadata = extractor.extract_metadata(sample_image_with_exif, tags=[])

        assert metadata == {}


# ============================================================================
# Test Class 3: ExifTask Execution
# ============================================================================


class TestExifTaskExecution:
    """Test ExifTask execution with real files."""

    @pytest.mark.asyncio
    async def test_task_with_exif_image(
        self,
        exif_task: ExifTask,
        sample_image_with_exif: str,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution with image containing EXIF data."""
        # Skip if exiftool not installed
        try:
            subprocess.run(
                ["exiftool", "-ver"], check=True, capture_output=True, timeout=5
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("ExifTool not installed")

        job = Job(
            job_id="test-job-1",
            task_type="exif",
            params={
                "input_paths": [sample_image_with_exif],
                "output_paths": [],
                "tags": [],  # Extract all
            },
        )

        params = ExifParams(**job.params)

        result = await exif_task.execute(job, params, mock_progress_callback)

        assert result["status"] == "ok"
        assert "task_output" in result
        assert result["task_output"]["total_files"] == 1
        assert len(result["task_output"]["files"]) == 1

        file_result = result["task_output"]["files"][0]
        assert file_result["file_path"] == sample_image_with_exif
        assert file_result["status"] in ["success", "no_metadata"]
        assert "metadata" in file_result

        # Progress callback should be called
        assert len(mock_progress_callback.calls) == 1
        assert 100 in mock_progress_callback.calls

    @pytest.mark.asyncio
    async def test_task_with_specific_tags(
        self,
        exif_task: ExifTask,
        sample_image_with_exif: str,
    ) -> None:
        """Test task execution with specific tags requested."""
        # Skip if exiftool not installed
        try:
            subprocess.run(
                ["exiftool", "-ver"], check=True, capture_output=True, timeout=5
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("ExifTool not installed")

        job = Job(
            job_id="test-job-2",
            task_type="exif",
            params={
                "input_paths": [sample_image_with_exif],
                "output_paths": [],
                "tags": ["Make", "Model", "Orientation"],
            },
        )

        params = ExifParams(**job.params)

        result = await exif_task.execute(job, params, None)

        assert result["status"] == "ok"
        assert result["task_output"]["tags_requested"] == ["Make", "Model", "Orientation"]

    @pytest.mark.asyncio
    async def test_task_with_nonexistent_file(
        self,
        exif_task: ExifTask,
    ) -> None:
        """Test task execution with non-existent file."""
        # Skip if exiftool not installed
        try:
            subprocess.run(
                ["exiftool", "-ver"], check=True, capture_output=True, timeout=5
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("ExifTool not installed")

        job = Job(
            job_id="test-job-3",
            task_type="exif",
            params={
                "input_paths": ["/nonexistent/file.jpg"],
                "output_paths": [],
                "tags": [],
            },
        )

        params = ExifParams(**job.params)

        result = await exif_task.execute(job, params, None)

        assert result["status"] == "error"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_task_without_exiftool(
        self,
        exif_task: ExifTask,
        sample_image_with_exif: str,
    ) -> None:
        """Test task execution when ExifTool is not available."""
        job = Job(
            job_id="test-job-4",
            task_type="exif",
            params={
                "input_paths": [sample_image_with_exif],
                "output_paths": [],
                "tags": [],
            },
        )

        params = ExifParams(**job.params)

        # Mock ExifTool not being available
        with patch(
            "cl_ml_tools.plugins.exif.task.MetadataExtractor",
            side_effect=RuntimeError("ExifTool is not installed or not found in PATH."),
        ):
            result = await exif_task.execute(job, params, None)

            assert result["status"] == "error"
            assert "ExifTool" in result["error"]

    @pytest.mark.asyncio
    async def test_task_with_multiple_files(
        self,
        exif_task: ExifTask,
        sample_image_with_exif: str,
        sample_image_without_exif: str,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution with multiple files."""
        # Skip if exiftool not installed
        try:
            subprocess.run(
                ["exiftool", "-ver"], check=True, capture_output=True, timeout=5
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("ExifTool not installed")

        job = Job(
            job_id="test-job-5",
            task_type="exif",
            params={
                "input_paths": [sample_image_with_exif, sample_image_without_exif],
                "output_paths": [],
                "tags": [],
            },
        )

        params = ExifParams(**job.params)

        result = await exif_task.execute(job, params, mock_progress_callback)

        assert result["status"] == "ok"
        assert result["task_output"]["total_files"] == 2
        assert len(result["task_output"]["files"]) == 2

        # Progress should be called twice (50%, 100%)
        assert len(mock_progress_callback.calls) == 2
        assert 50 in mock_progress_callback.calls
        assert 100 in mock_progress_callback.calls


# ============================================================================
# Test Class 4: Task Properties
# ============================================================================


class TestExifTaskProperties:
    """Test ExifTask properties and schema."""

    def test_task_type(self, exif_task: ExifTask) -> None:
        """Test task_type property."""
        assert exif_task.task_type == "exif"

    def test_get_schema(self, exif_task: ExifTask) -> None:
        """Test get_schema method."""
        schema = exif_task.get_schema()

        assert schema == ExifParams
