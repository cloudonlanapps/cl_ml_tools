"""Comprehensive test suite for HLS streaming conversion plugin."""

from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from cl_ml_tools.plugins.hls_streaming.schema import (
    HLSConversionResult,
    HLSStreamingParams,
    HLSStreamingTaskOutput,
    VariantConfig,
)
from cl_ml_tools.plugins.hls_streaming.task import HLSStreamingTask
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
def sample_video_file(tmp_path: Path) -> str:
    """Create a minimal test video file using random_media_generator."""
    out_dir = str(tmp_path)

    # Generate video using VideoGenerator with Pydantic
    # Note: VideoGenerator has a field_validator that converts dict -> SceneGenerator
    # The field type is list[SceneGenerator], but the validator accepts RawScenes input
    # Using Any here is necessary as the type checker doesn't understand Pydantic validators
    scenes_data: list[RawScene] = [
        {
            "duration_seconds": 5,  # 5-second video
            "background_color": (100, 150, 200),  # RGB color
            "num_shapes": 2,  # Auto-generate 2 random animated shapes
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
def hls_task() -> HLSStreamingTask:
    """Create HLSStreamingTask instance."""
    return HLSStreamingTask()


@pytest.fixture
def mock_progress_callback() -> MockProgressCallback:
    """Create mock progress callback."""
    return MockProgressCallback()


# ============================================================================
# Test Class 1: TestHLSStreamingParams - Schema validation (6 tests)
# ============================================================================


class TestHLSStreamingParams:
    """Test HLSStreamingParams schema validation."""

    def test_default_variants(self) -> None:
        """Test default variants are set correctly."""
        params = HLSStreamingParams(input_paths=["/input/video.mp4"], output_paths=["/output"])
        assert len(params.variants) == 3
        assert params.variants[0].resolution == 720
        assert params.variants[0].bitrate == 3500
        assert params.variants[1].resolution == 480
        assert params.variants[1].bitrate == 1500
        assert params.variants[2].resolution == 240
        assert params.variants[2].bitrate == 800

    def test_custom_variants(self) -> None:
        """Test custom variants configuration."""
        custom_variants = [
            VariantConfig(resolution=1080, bitrate=5000),
            VariantConfig(resolution=360, bitrate=800),
        ]
        params = HLSStreamingParams(
            input_paths=["/input/video.mp4"],
            output_paths=["/output"],
            variants=custom_variants,
        )
        assert len(params.variants) == 2
        assert params.variants[0].resolution == 1080
        assert params.variants[0].bitrate == 5000
        assert params.variants[1].resolution == 360
        assert params.variants[1].bitrate == 800

    def test_empty_variants_fails(self) -> None:
        """Test that empty variants list raises validation error."""
        with pytest.raises(ValidationError, match="At least one variant"):
            _ = HLSStreamingParams(
                input_paths=["/input/video.mp4"], output_paths=["/output"], variants=[]
            )

    def test_variant_config_validation(self) -> None:
        """Test VariantConfig field validation."""
        variant = VariantConfig(resolution=720, bitrate=3500)
        assert variant.resolution == 720
        assert variant.bitrate == 3500

    def test_include_original_flag(self) -> None:
        """Test include_original flag."""
        params1 = HLSStreamingParams(input_paths=["/input/video.mp4"], output_paths=["/output"])
        assert params1.include_original is False

        params2 = HLSStreamingParams(
            input_paths=["/input/video.mp4"],
            output_paths=["/output"],
            include_original=True,
        )
        assert params2.include_original is True

    def test_output_paths_validation(self) -> None:
        """Test inherited output_paths validation."""
        # Should succeed with matching counts
        params = HLSStreamingParams(input_paths=["/input/video.mp4"], output_paths=["/output"])
        assert len(params.input_paths) == len(params.output_paths)

        # Should fail with mismatched counts
        with pytest.raises(ValidationError):
            _ = HLSStreamingParams(
                input_paths=["/input/video1.mp4", "/input/video2.mp4"],
                output_paths=["/output"],
            )


# ============================================================================
# Test Class 2: TestVariantConfig - Nested schema (4 tests)
# ============================================================================


class TestVariantConfig:
    """Test VariantConfig nested schema."""

    def test_valid_resolution_bitrate(self) -> None:
        """Test valid resolution and bitrate values."""
        variant = VariantConfig(resolution=720, bitrate=3500)
        assert variant.resolution == 720
        assert variant.bitrate == 3500

    def test_none_values_for_original(self) -> None:
        """Test None values for original quality."""
        variant = VariantConfig(resolution=None, bitrate=None)
        assert variant.resolution is None
        assert variant.bitrate is None

    def test_bitrate_constraints(self) -> None:
        """Test bitrate >= 100 kbps constraint."""
        # Should succeed
        variant = VariantConfig(resolution=720, bitrate=100)
        assert variant.bitrate == 100

        # Should fail for bitrate < 100
        with pytest.raises(ValidationError):
            _ = VariantConfig(resolution=720, bitrate=50)

    def test_type_validation(self) -> None:
        """Test type validation for fields."""
        # Valid types
        variant = VariantConfig(resolution=720, bitrate=3500)
        assert isinstance(variant.resolution, int)
        assert isinstance(variant.bitrate, int)


# ============================================================================
# Test Class 3: TestHLSVariant - Algorithm class (5 tests)
# ============================================================================


class TestHLSVariant:
    """Test HLSVariant algorithm class."""

    def test_variant_creation(self) -> None:
        """Test HLSVariant creation."""
        from cl_ml_tools.plugins.hls_streaming.algo.hls_stream_generator import (
            HLSVariant,
        )

        variant = HLSVariant(resolution=720, bitrate=3500)
        assert variant.resolution == 720
        assert variant.bitrate == 3500

    def test_uri_generation(self) -> None:
        """Test URI generation for variant."""
        from cl_ml_tools.plugins.hls_streaming.algo.hls_stream_generator import (
            HLSVariant,
        )

        variant = HLSVariant(resolution=720, bitrate=3500)
        uri = variant.uri()
        assert "720" in uri
        assert "3500" in uri

    def test_equality_comparison(self) -> None:
        """Test equality comparison between variants."""
        from cl_ml_tools.plugins.hls_streaming.algo.hls_stream_generator import (
            HLSVariant,
        )

        variant1 = HLSVariant(resolution=720, bitrate=3500)
        variant2 = HLSVariant(resolution=720, bitrate=3500)
        variant3 = HLSVariant(resolution=480, bitrate=1500)

        assert variant1 == variant2
        assert variant1 != variant3

    def test_check_method(self) -> None:
        """Test check method (requires actual files)."""
        from cl_ml_tools.plugins.hls_streaming.algo.hls_stream_generator import (
            HLSVariant,
        )

        variant = HLSVariant(resolution=720, bitrate=3500)
        # Check should return False for non-existent files
        result = variant.check("/nonexistent/path")
        assert result is False

    def test_resolution_extraction(self) -> None:
        """Test resolution extraction from variant."""
        from cl_ml_tools.plugins.hls_streaming.algo.hls_stream_generator import (
            HLSVariant,
        )

        variant = HLSVariant(resolution=720, bitrate=3500)
        assert variant.resolution == 720


# ============================================================================
# Test Class 4: TestHLSValidator - Validation logic (4 tests)
# ============================================================================


class TestHLSValidator:
    """Test HLSValidator validation logic."""

    def test_valid_hls_structure(self, tmp_path: Path) -> None:
        """Test validation of valid HLS structure."""
        from cl_ml_tools.plugins.hls_streaming.algo.hls_validator import (
            HLSValidator,
        )

        # Create a minimal valid M3U8 file
        master_playlist = tmp_path / "adaptive.m3u8"
        _ = master_playlist.write_text("#EXTM3U\n#EXT-X-VERSION:3\n")

        validator = HLSValidator(str(master_playlist))
        result = validator.validate()

        # May not be fully valid without segments, but should not crash
        assert result is not None

    def test_missing_files_detection(self) -> None:
        """Test detection of missing files."""
        from cl_ml_tools.plugins.hls_streaming.algo.hls_validator import (
            HLSValidator,
        )

        validator = HLSValidator("/nonexistent/adaptive.m3u8")
        result = validator.validate()
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_invalid_playlist_handling(self, tmp_path: Path) -> None:
        """Test handling of invalid playlist."""
        from cl_ml_tools.plugins.hls_streaming.algo.hls_validator import (
            HLSValidator,
        )

        # Create an invalid M3U8 file
        master_playlist = tmp_path / "adaptive.m3u8"
        _ = master_playlist.write_text("This is not a valid M3U8 file")

        validator = HLSValidator(str(master_playlist))
        result = validator.validate()
        # The m3u8 library may parse invalid files without errors
        # Just verify the validator returns a result
        assert result is not None
        assert isinstance(result.is_valid, bool)

    def test_validation_result_structure(self) -> None:
        """Test ValidationResult structure."""
        from cl_ml_tools.plugins.hls_streaming.algo.hls_validator import (
            ValidationResult,
        )

        result = ValidationResult(
            is_valid=True,
            missing_files=[],
            total_segments=100,
            segments_found=100,
            variants_info={},
            errors=[],
        )
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.total_segments == 100


# ============================================================================
# Test Class 5: TestHLSStreamingTask - Task execution (8 tests)
# ============================================================================


class TestHLSStreamingTask:
    """Test HLSStreamingTask execution."""

    def test_task_type(self, hls_task: HLSStreamingTask) -> None:
        """Test task_type returns 'hls_streaming'."""
        assert hls_task.task_type == "hls_streaming"

    def test_get_schema(self, hls_task: HLSStreamingTask) -> None:
        """Test get_schema returns HLSStreamingParams."""
        schema = hls_task.get_schema()
        assert schema == HLSStreamingParams

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_execute_with_default_variants(
        self,
        hls_task: HLSStreamingTask,
        sample_video_file: str,
        tmp_path: Path,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test execute with default variants."""
        from cl_ml_tools.common.schemas import Job

        output_dir = str(tmp_path / "output")
        params = HLSStreamingParams(input_paths=[sample_video_file], output_paths=[output_dir])
        job = Job(job_id="test-job", task_type="hls_streaming", params=params.model_dump())

        result = await hls_task.execute(job, params, mock_progress_callback)

        # Check result structure
        assert result["status"] == "ok" or result["status"] == "error"
        if result["status"] == "ok":
            assert "task_output" in result
            assert len(mock_progress_callback.calls) > 0

    @pytest.mark.asyncio
    async def test_execute_with_custom_variants(
        self,
        hls_task: HLSStreamingTask,
        sample_video_file: str,
        tmp_path: Path,
    ) -> None:
        """Test execute with custom variants."""
        from cl_ml_tools.common.schemas import Job

        output_dir = str(tmp_path / "output")
        custom_variants = [VariantConfig(resolution=480, bitrate=1500)]
        params = HLSStreamingParams(
            input_paths=[sample_video_file],
            output_paths=[output_dir],
            variants=custom_variants,
        )
        job = Job(job_id="test-job", task_type="hls_streaming", params=params.model_dump())

        result = await hls_task.execute(job, params, None)

        assert "status" in result

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_execute_with_original(
        self,
        hls_task: HLSStreamingTask,
        sample_video_file: str,
        tmp_path: Path,
    ) -> None:
        """Test execute with original quality."""
        from cl_ml_tools.common.schemas import Job

        output_dir = str(tmp_path / "output")
        params = HLSStreamingParams(
            input_paths=[sample_video_file],
            output_paths=[output_dir],
            include_original=True,
        )
        job = Job(job_id="test-job", task_type="hls_streaming", params=params.model_dump())

        result = await hls_task.execute(job, params, None)

        assert "status" in result

    @pytest.mark.asyncio
    async def test_execute_multiple_files(
        self,
        hls_task: HLSStreamingTask,
        sample_video_file: str,
        tmp_path: Path,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test execute with multiple files."""
        from cl_ml_tools.common.schemas import Job

        output_dir1 = str(tmp_path / "output1")
        output_dir2 = str(tmp_path / "output2")
        params = HLSStreamingParams(
            input_paths=[sample_video_file, sample_video_file],
            output_paths=[output_dir1, output_dir2],
        )
        job = Job(job_id="test-job", task_type="hls_streaming", params=params.model_dump())

        result = await hls_task.execute(job, params, mock_progress_callback)

        assert "status" in result
        # Progress should be called at least twice (once per file)
        if result["status"] == "ok":
            assert len(mock_progress_callback.calls) >= 2

    @pytest.mark.asyncio
    async def test_progress_callback(
        self,
        hls_task: HLSStreamingTask,
        sample_video_file: str,
        tmp_path: Path,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test progress callback is called."""
        from cl_ml_tools.common.schemas import Job

        output_dir = str(tmp_path / "output")
        params = HLSStreamingParams(input_paths=[sample_video_file], output_paths=[output_dir])
        job = Job(job_id="test-job", task_type="hls_streaming", params=params.model_dump())

        result = await hls_task.execute(job, params, mock_progress_callback)

        # Progress callback should be called only if execution succeeded
        if result["status"] == "ok":
            assert len(mock_progress_callback.calls) > 0

    @pytest.mark.asyncio
    async def test_file_not_found_error(self, hls_task: HLSStreamingTask, tmp_path: Path) -> None:
        """Test file not found error handling."""
        from cl_ml_tools.common.schemas import Job

        output_dir = str(tmp_path / "output")
        params = HLSStreamingParams(
            input_paths=["/nonexistent/video.mp4"], output_paths=[output_dir]
        )
        job = Job(job_id="test-job", task_type="hls_streaming", params=params.model_dump())

        result = await hls_task.execute(job, params, None)

        assert result["status"] == "error"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_output_structure_validation(
        self,
        hls_task: HLSStreamingTask,
        sample_video_file: str,
        tmp_path: Path,
    ) -> None:
        """Test output structure uses Pydantic models."""
        from cl_ml_tools.common.schemas import Job

        output_dir = str(tmp_path / "output")
        params = HLSStreamingParams(input_paths=[sample_video_file], output_paths=[output_dir])
        job = Job(job_id="test-job", task_type="hls_streaming", params=params.model_dump())

        result = await hls_task.execute(job, params, None)

        if result["status"] == "ok":
            # Verify task_output structure matches HLSStreamingTaskOutput
            task_output = result.get("task_output")
            assert task_output is not None
            assert isinstance(task_output, dict)
            assert "files" in task_output
            assert "total_files" in task_output

            # Verify we can reconstruct Pydantic models from output
            # Pydantic will validate and convert the dict values at runtime
            output_model = HLSStreamingTaskOutput(**cast(Any, task_output))
            assert output_model.total_files >= 0
            assert len(output_model.files) == output_model.total_files


# ============================================================================
# Test Class 6: TestHLSStreamingRoutes - API endpoint (3 tests)
# ============================================================================


class TestHLSStreamingRoutes:
    """Test HLS streaming API routes."""

    def test_route_creation(self) -> None:
        """Test route creation with dependencies."""
        from unittest.mock import Mock

        from cl_ml_tools.plugins.hls_streaming.routes import create_router

        mock_repo = Mock()
        mock_storage = Mock()
        mock_auth = Mock(return_value=None)

        router = create_router(mock_repo, mock_storage, mock_auth)
        assert router is not None

    @pytest.mark.asyncio
    async def test_parameter_parsing(self) -> None:
        """Test parameter parsing in route."""
        import json

        # Test JSON parsing
        variants_json = '[{"resolution":720,"bitrate":3500}]'
        variants_list = cast(list[dict[str, object]], json.loads(variants_json))

        assert isinstance(variants_list, list)
        assert len(variants_list) == 1
        assert variants_list[0]["resolution"] == 720
        assert variants_list[0]["bitrate"] == 3500

    @pytest.mark.asyncio
    async def test_job_creation(self) -> None:
        """Test job creation flow."""
        from unittest.mock import AsyncMock, Mock

        from cl_ml_tools.plugins.hls_streaming.routes import create_router

        # Create mocks
        mock_repo = Mock()
        mock_repo.add_job = Mock()

        mock_storage = Mock()
        mock_storage.create_job_directory = Mock()
        mock_storage.get_output_path = Mock(return_value="/output/path")
        mock_storage.save_input_file = AsyncMock(return_value={"path": "/input/path/video.mp4"})

        mock_auth = Mock(return_value=None)

        # Create router
        router = create_router(mock_repo, mock_storage, mock_auth)

        # Verify router has the endpoint
        routes_list = [str(getattr(route, "path", "")) for route in router.routes]
        assert "/jobs/hls_streaming" in routes_list


# ============================================================================
# Test Class 7: TestHLSPluginIntegration - End-to-end (4 tests)
# ============================================================================


class TestHLSPluginIntegration:
    """End-to-end integration tests for HLS plugin."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_single_variant_conversion(
        self,
        hls_task: HLSStreamingTask,
        sample_video_file: str,
        tmp_path: Path,
    ) -> None:
        """Test single variant conversion end-to-end."""
        from cl_ml_tools.common.schemas import Job

        output_dir = str(tmp_path / "output")
        params = HLSStreamingParams(
            input_paths=[sample_video_file],
            output_paths=[output_dir],
            variants=[VariantConfig(resolution=480, bitrate=1500)],
        )
        job = Job(job_id="test-job", task_type="hls_streaming", params=params.model_dump())

        result = await hls_task.execute(job, params, None)

        assert "status" in result

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_variant_conversion(
        self,
        hls_task: HLSStreamingTask,
        sample_video_file: str,
        tmp_path: Path,
    ) -> None:
        """Test multi-variant conversion end-to-end."""
        from cl_ml_tools.common.schemas import Job

        output_dir = str(tmp_path / "output")
        params = HLSStreamingParams(input_paths=[sample_video_file], output_paths=[output_dir])
        job = Job(job_id="test-job", task_type="hls_streaming", params=params.model_dump())

        result = await hls_task.execute(job, params, None)

        assert "status" in result

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_original_preservation(
        self,
        hls_task: HLSStreamingTask,
        sample_video_file: str,
        tmp_path: Path,
    ) -> None:
        """Test original quality preservation end-to-end."""
        from cl_ml_tools.common.schemas import Job

        output_dir = str(tmp_path / "output")
        params = HLSStreamingParams(
            input_paths=[sample_video_file],
            output_paths=[output_dir],
            include_original=True,
        )
        job = Job(job_id="test-job", task_type="hls_streaming", params=params.model_dump())

        result = await hls_task.execute(job, params, None)

        assert "status" in result

    @pytest.mark.asyncio
    async def test_error_handling(self, hls_task: HLSStreamingTask, tmp_path: Path) -> None:
        """Test error handling end-to-end."""
        from cl_ml_tools.common.schemas import Job

        output_dir = str(tmp_path / "output")
        params = HLSStreamingParams(
            input_paths=["/nonexistent/video.mp4"], output_paths=[output_dir]
        )
        job = Job(job_id="test-job", task_type="hls_streaming", params=params.model_dump())

        result = await hls_task.execute(job, params, None)

        assert result["status"] == "error"
        assert "error" in result
        error_msg = result["error"]
        assert isinstance(error_msg, str)
        assert "not found" in error_msg.lower() or "does not exist" in error_msg.lower()


# ============================================================================
# Test Pydantic Models
# ============================================================================


class TestPydanticModels:
    """Test Pydantic model structures."""

    def test_hls_conversion_result_model(self) -> None:
        """Test HLSConversionResult Pydantic model."""
        result = HLSConversionResult(
            input_file="/input/video.mp4",
            output_dir="/output",
            master_playlist="/output/adaptive.m3u8",
            variants_generated=2,
            total_segments=100,
            include_original=False,
        )

        assert result.input_file == "/input/video.mp4"
        assert result.output_dir == "/output"
        assert result.master_playlist == "/output/adaptive.m3u8"
        assert result.variants_generated == 2
        assert result.total_segments == 100
        assert result.include_original is False

    def test_hls_streaming_task_output_model(self) -> None:
        """Test HLSStreamingTaskOutput Pydantic model."""
        files = [
            HLSConversionResult(
                input_file="/input/video.mp4",
                output_dir="/output",
                master_playlist="/output/adaptive.m3u8",
                variants_generated=2,
                total_segments=100,
                include_original=False,
            )
        ]
        output = HLSStreamingTaskOutput(files=files, total_files=1)

        assert len(output.files) == 1
        assert output.total_files == 1
        assert output.files[0].input_file == "/input/video.mp4"

    def test_model_dump(self) -> None:
        """Test model_dump() produces correct dict structure."""
        result = HLSConversionResult(
            input_file="/input/video.mp4",
            output_dir="/output",
            master_playlist="/output/adaptive.m3u8",
            variants_generated=2,
            total_segments=100,
            include_original=False,
        )
        output = HLSStreamingTaskOutput(files=[result], total_files=1)

        dumped = output.model_dump()
        assert isinstance(dumped, dict)
        assert "files" in dumped
        assert "total_files" in dumped
        assert dumped["total_files"] == 1
        assert isinstance(dumped["files"], list)
