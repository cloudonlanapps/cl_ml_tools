"""Comprehensive test suite for image/video resize plugin."""

from pathlib import Path
from typing import cast

import pytest
from PIL import Image

from cl_ml_tools.plugins.media_resize.schema import MediaResizeParams
from cl_ml_tools.plugins.media_resize.task import MediaResizeTask
from cl_ml_tools.utils.random_media_generator.frame_generator import FrameGenerator
from cl_ml_tools.utils.random_media_generator.image_generator import ImageGenerator
from cl_ml_tools.utils.random_media_generator.scene_generator import SceneGenerator
from cl_ml_tools.utils.random_media_generator.video_generator import RawScene, VideoGenerator


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
def sample_image_file(tmp_path: Path) -> str:
    """Create a test image using random_media_generator."""
    out_dir = str(tmp_path)

    frame = FrameGenerator(
        background_color=(200, 200, 200),
        num_shapes=3,
    )

    image_gen = ImageGenerator(
        out_dir=out_dir,
        MIMEType="image/jpeg",
        fileName="test_image",
        width=800,
        height=600,
        frame=frame,
    )
    image_gen.generate()

    return image_gen.filepath


@pytest.fixture
def sample_video_file(tmp_path: Path) -> str:
    """Create a test video using random_media_generator."""
    out_dir = str(tmp_path)

    scenes_data: list[RawScene] = [
        {
            "duration_seconds": 3,
            "background_color": (100, 150, 200),
            "num_shapes": 2,
        }
    ]
    video_gen = VideoGenerator(
        out_dir=out_dir,
        MIMEType="video/mp4",
        fileName="test_video",
        width=640,
        height=480,
        fps=30,
        scenes=[SceneGenerator.model_validate(item) for item in scenes_data],
    )
    video_gen.generate()

    return video_gen.filepath


@pytest.fixture
def resize_task() -> MediaResizeTask:
    """Create MediaResizeTask instance."""
    return MediaResizeTask()


@pytest.fixture
def mock_progress_callback() -> MockProgressCallback:
    """Create mock progress callback."""
    return MockProgressCallback()


# ============================================================================
# Test Class 1: Schema Validation
# ============================================================================


class TestMediaResizeParams:
    """Test MediaResizeParams schema validation."""

    def test_default_dimensions(self) -> None:
        """Test default width/height are None."""
        params = MediaResizeParams(
            input_paths=["/input/image.jpg"], output_paths=["/output/image.jpg"]
        )
        assert params.width is None
        assert params.height is None
        assert params.maintain_aspect_ratio is True

    def test_custom_dimensions(self) -> None:
        """Test custom width/height."""
        params = MediaResizeParams(
            input_paths=["/input/image.jpg"],
            output_paths=["/output/image.jpg"],
            width=512,
            height=384,
        )
        assert params.width == 512
        assert params.height == 384

    def test_width_only(self) -> None:
        """Test width-only specification."""
        params = MediaResizeParams(
            input_paths=["/input/image.jpg"],
            output_paths=["/output/image.jpg"],
            width=512,
        )
        assert params.width == 512
        assert params.height is None


# ============================================================================
# Test Class 2: Image Resize Functionality
# ============================================================================


class TestImageResize:
    """Test image resize functionality."""

    @pytest.mark.asyncio
    async def test_image_resize_default_dimensions(
        self,
        sample_image_file: str,
        tmp_path: Path,
        resize_task: MediaResizeTask,
    ) -> None:
        """Test image resize with default 256x256."""
        output_path = str(tmp_path / "output.jpg")

        params = MediaResizeParams(
            input_paths=[sample_image_file],
            output_paths=[output_path],
        )

        result = await resize_task.execute(job=cast(dict, {}), params=params)

        assert result["status"] == "ok"
        assert Path(output_path).exists()

        # Verify dimensions
        with Image.open(output_path) as img:
            assert max(img.size) <= 256

    @pytest.mark.asyncio
    async def test_image_resize_width_only(
        self,
        sample_image_file: str,
        tmp_path: Path,
        resize_task: MediaResizeTask,
    ) -> None:
        """Test image resize with width only (maintain aspect ratio)."""
        output_path = str(tmp_path / "output.jpg")

        params = MediaResizeParams(
            input_paths=[sample_image_file],
            output_paths=[output_path],
            width=512,
        )

        result = await resize_task.execute(job=cast(dict, {}), params=params)

        assert result["status"] == "ok"
        task_output = result["task_output"]
        assert task_output["media_types"] == ["image"]

        # Verify dimensions
        with Image.open(output_path) as img:
            assert img.size[0] <= 512  # Width should be <= 512

    @pytest.mark.asyncio
    async def test_image_resize_both_dimensions(
        self,
        sample_image_file: str,
        tmp_path: Path,
        resize_task: MediaResizeTask,
    ) -> None:
        """Test image resize with both width and height (fit in box)."""
        output_path = str(tmp_path / "output.jpg")

        params = MediaResizeParams(
            input_paths=[sample_image_file],
            output_paths=[output_path],
            width=640,
            height=480,
        )

        result = await resize_task.execute(job=cast(dict, {}), params=params)

        assert result["status"] == "ok"

        # Verify fits within box
        with Image.open(output_path) as img:
            assert img.size[0] <= 640
            assert img.size[1] <= 480


# ============================================================================
# Test Class 3: Video Resize Functionality
# ============================================================================


class TestVideoResize:
    """Test video resize functionality."""

    @pytest.mark.asyncio
    async def test_video_resize_default_dimensions(
        self,
        sample_video_file: str,
        tmp_path: Path,
        resize_task: MediaResizeTask,
    ) -> None:
        """Test video resize (thumbnail) with default 256x256."""
        output_path = str(tmp_path / "thumbnail.jpg")

        params = MediaResizeParams(
            input_paths=[sample_video_file],
            output_paths=[output_path],
        )

        result = await resize_task.execute(job=cast(dict, {}), params=params)

        assert result["status"] == "ok"
        task_output = result["task_output"]
        assert task_output["media_types"] == ["video"]
        assert Path(output_path).exists()

        # Verify output is an image
        with Image.open(output_path) as img:
            assert img.size[0] > 0
            assert img.size[1] > 0

    @pytest.mark.asyncio
    async def test_video_resize_custom_width(
        self,
        sample_video_file: str,
        tmp_path: Path,
        resize_task: MediaResizeTask,
    ) -> None:
        """Test video resize with custom width."""
        output_path = str(tmp_path / "thumbnail.jpg")

        params = MediaResizeParams(
            input_paths=[sample_video_file],
            output_paths=[output_path],
            width=512,
        )

        result = await resize_task.execute(job=cast(dict, {}), params=params)

        assert result["status"] == "ok"
        assert Path(output_path).exists()


# ============================================================================
# Test Class 4: Mixed Media Types
# ============================================================================


class TestMixedMedia:
    """Test handling mixed image and video inputs."""

    @pytest.mark.asyncio
    async def test_mixed_media_batch(
        self,
        sample_image_file: str,
        sample_video_file: str,
        tmp_path: Path,
        resize_task: MediaResizeTask,
    ) -> None:
        """Test processing both images and videos in one batch."""
        output_image = str(tmp_path / "output_image.jpg")
        output_video = str(tmp_path / "output_video.jpg")

        params = MediaResizeParams(
            input_paths=[sample_image_file, sample_video_file],
            output_paths=[output_image, output_video],
            width=256,
            height=256,
        )

        result = await resize_task.execute(job=cast(dict, {}), params=params)

        assert result["status"] == "ok"
        task_output = result["task_output"]
        assert task_output["media_types"] == ["image", "video"]
        assert len(task_output["processed_files"]) == 2
        assert Path(output_image).exists()
        assert Path(output_video).exists()


# ============================================================================
# Test Class 5: Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_file_not_found(
        self,
        tmp_path: Path,
        resize_task: MediaResizeTask,
    ) -> None:
        """Test error when input file doesn't exist."""
        params = MediaResizeParams(
            input_paths=["/nonexistent/file.jpg"],
            output_paths=[str(tmp_path / "output.jpg")],
        )

        result = await resize_task.execute(job=cast(dict, {}), params=params)

        assert result["status"] == "error"
        assert "not found" in result["error"].lower()
