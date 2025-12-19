"""Unit and integration tests for HLS streaming plugin.

Tests schema validation, HLS playlist generation, variant configuration, task execution, routes, and full job lifecycle.
Requires FFmpeg to be installed.
"""

from pathlib import Path

import pytest

from cl_ml_tools.plugins.hls_streaming.schema import (
    HLSStreamingOutput,
    HLSStreamingParams,
    VariantConfig,
)
from cl_ml_tools.plugins.hls_streaming.task import HLSStreamingTask


# ============================================================================
# SCHEMA TESTS
# ============================================================================


def test_variant_config_schema_validation():
    """Test VariantConfig schema validates correctly."""
    variant = VariantConfig(resolution=720, bitrate=3500)

    assert variant.resolution == 720
    assert variant.bitrate == 3500


def test_variant_config_defaults():
    """Test VariantConfig default values."""
    variant = VariantConfig()

    assert variant.resolution is None
    assert variant.bitrate is None


def test_variant_config_bitrate_validation():
    """Test VariantConfig validates bitrate minimum."""
    # Valid bitrate
    variant = VariantConfig(resolution=480, bitrate=1000)
    assert variant.bitrate == 1000

    # Invalid bitrate (too low)
    with pytest.raises(ValueError):
        _ = VariantConfig(resolution=480, bitrate=50)


def test_hls_streaming_params_schema_validation():
    """Test HLSStreamingParams schema validates correctly."""
    params = HLSStreamingParams(
        input_path="/path/to/input.mp4",
        output_path="output/hls",
        variants=[
            VariantConfig(resolution=720, bitrate=3500),
            VariantConfig(resolution=480, bitrate=1500),
        ],
        include_original=True,
    )

    assert params.input_path == "/path/to/input.mp4"
    assert params.output_path == "output/hls"
    assert len(params.variants) == 2
    assert params.include_original is True


def test_hls_streaming_params_defaults():
    """Test HLSStreamingParams has correct default values."""
    params = HLSStreamingParams(
        input_path="/path/to/input.mp4",
        output_path="output/hls",
    )

    # Default variants: 720p, 480p, 240p
    assert len(params.variants) == 3
    assert params.variants[0].resolution == 720
    assert params.variants[1].resolution == 480
    assert params.variants[2].resolution == 240
    assert params.include_original is False


def test_hls_streaming_params_empty_variants_validation():
    """Test HLSStreamingParams rejects empty variants list."""
    with pytest.raises(ValueError):
        _ = HLSStreamingParams(
            input_path="/path/to/input.mp4",
            output_path="output/hls",
            variants=[],
        )


def test_hls_streaming_params_custom_variants():
    """Test HLSStreamingParams with custom variants."""
    params = HLSStreamingParams(
        input_path="/path/to/input.mp4",
        output_path="output/hls",
        variants=[
            VariantConfig(resolution=1080, bitrate=5000),
            VariantConfig(resolution=360, bitrate=800),
        ],
    )

    assert len(params.variants) == 2
    assert params.variants[0].resolution == 1080
    assert params.variants[1].resolution == 360


def test_hls_streaming_output_schema_validation():
    """Test HLSStreamingOutput schema validates correctly."""
    output = HLSStreamingOutput(
        master_playlist="output/hls/master.m3u8",
        variants_generated=3,
        total_segments=120,
        include_original=False,
    )

    assert output.master_playlist == "output/hls/master.m3u8"
    assert output.variants_generated == 3
    assert output.total_segments == 120
    assert output.include_original is False


# ============================================================================
# TASK TESTS
# ============================================================================


@pytest.mark.requires_ffmpeg
@pytest.mark.asyncio
async def test_hls_streaming_task_run_success(sample_video_path: Path, tmp_path: Path):
    """Test HLSStreamingTask execution success."""
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(sample_video_path.read_bytes())

    params = HLSStreamingParams(
        input_path=str(input_path),
        output_path="output/hls",
        variants=[
            VariantConfig(resolution=480, bitrate=1500),
            VariantConfig(resolution=240, bitrate=800),
        ],
    )

    task = HLSStreamingTask()
    job_id = "test-job-123"

    class MockStorage:
        def resolve_path(self, job_id: str, relative_path: str) -> Path:
            return tmp_path / job_id / relative_path

        def allocate_path(self, job_id: str, relative_path: str) -> str:
            output_path = tmp_path / "output" / "hls" / Path(relative_path).name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return str(output_path)

    storage = MockStorage()

    output = await task.run(job_id, params, storage)

    assert isinstance(output, HLSStreamingOutput)
    assert output.variants_generated > 0
    assert output.total_segments > 0


@pytest.mark.requires_ffmpeg
@pytest.mark.asyncio
async def test_hls_streaming_task_creates_master_playlist(sample_video_path: Path, tmp_path: Path):
    """Test HLSStreamingTask creates master playlist."""
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(sample_video_path.read_bytes())

    params = HLSStreamingParams(
        input_path=str(input_path),
        output_path="output/hls",
        variants=[VariantConfig(resolution=480, bitrate=1500)],
    )

    task = HLSStreamingTask()
    job_id = "test-job-456"

    class MockStorage:
        def resolve_path(self, job_id: str, relative_path: str) -> Path:
            return tmp_path / job_id / relative_path

        def allocate_path(self, job_id: str, relative_path: str) -> str:
            output_path = tmp_path / "output" / "hls" / Path(relative_path).name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return str(output_path)

    storage = MockStorage()

    output = await task.run(job_id, params, storage)

    # Master playlist should be created
    assert "adaptive.m3u8" in output.master_playlist


@pytest.mark.requires_ffmpeg
@pytest.mark.asyncio
async def test_hls_streaming_task_run_file_not_found(tmp_path: Path):
    """Test HLSStreamingTask raises FileNotFoundError for missing input."""
    params = HLSStreamingParams(
        input_path="/nonexistent/file.mp4",
        output_path="output/hls",
    )

    task = HLSStreamingTask()
    job_id = "test-job-789"

    class MockStorage:
        def resolve_path(self, job_id: str, relative_path: str) -> Path:
            return tmp_path / job_id / relative_path

        def allocate_path(self, job_id: str, relative_path: str) -> str:
            output_path = tmp_path / "output" / "hls" / Path(relative_path).name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return str(output_path)

    storage = MockStorage()

    with pytest.raises(FileNotFoundError):
        await task.run(job_id, params, storage)


@pytest.mark.requires_ffmpeg
@pytest.mark.asyncio
async def test_hls_streaming_task_progress_callback(sample_video_path: Path, tmp_path: Path):
    """Test HLSStreamingTask calls progress callback."""
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(sample_video_path.read_bytes())

    params = HLSStreamingParams(
        input_path=str(input_path),
        output_path="output/hls",
        variants=[VariantConfig(resolution=240, bitrate=800)],
    )

    task = HLSStreamingTask()
    job_id = "test-job-progress"

    class MockStorage:
        def resolve_path(self, job_id: str, relative_path: str) -> Path:
            return tmp_path / job_id / relative_path

        def allocate_path(self, job_id: str, relative_path: str) -> str:
            output_path = tmp_path / "output" / "hls" / Path(relative_path).name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return str(output_path)

    storage = MockStorage()

    progress_values = []

    def progress_callback(progress: int):
        progress_values.append(progress)

    await task.run(job_id, params, storage, progress_callback)

    # Should have some progress updates
    assert len(progress_values) > 0
    assert 100 in progress_values


@pytest.mark.requires_ffmpeg
@pytest.mark.asyncio
async def test_hls_streaming_task_audio_only_failure(tmp_path: Path):
    """Test HLSStreamingTask rejects audio-only input."""
    import subprocess

    # Generate dummy audio file
    input_path = tmp_path / "audio_only.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo",
            "-t",
            "5",
            "-c:a",
            "aac",
            str(input_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    params = HLSStreamingParams(
        input_path=str(input_path),
        output_path="output/hls",
        variants=[VariantConfig(resolution=240, bitrate=800)],
    )

    task = HLSStreamingTask()
    job_id = "test-job-audio-only"

    class MockStorage:
        def resolve_path(self, job_id: str, relative_path: str) -> Path:
            return tmp_path / relative_path

        def allocate_path(self, job_id: str, relative_path: str) -> str:
            output_path = tmp_path / "output" / "hls" / Path(relative_path).name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return str(output_path)

    storage = MockStorage()

    with pytest.raises(ValueError, match="does not contain a video stream"):
        await task.run(job_id, params, storage)



@pytest.mark.requires_ffmpeg
@pytest.mark.asyncio
async def test_hls_streaming_task_with_original(sample_video_path: Path, tmp_path: Path):
    """Test HLSStreamingTask with include_original flag."""
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(sample_video_path.read_bytes())

    params = HLSStreamingParams(
        input_path=str(input_path),
        output_path="output/hls",
        variants=[VariantConfig(resolution=480, bitrate=1500)],
        include_original=True,
    )

    task = HLSStreamingTask()
    job_id = "test-job-original"

    class MockStorage:
        def resolve_path(self, job_id: str, relative_path: str) -> Path:
            return tmp_path / job_id / relative_path

        def allocate_path(self, job_id: str, relative_path: str) -> str:
            output_path = tmp_path / "output" / "hls" / Path(relative_path).name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return str(output_path)

    storage = MockStorage()

    output = await task.run(job_id, params, storage)

    assert output.include_original is True
    # Should have generated variants + original
    assert output.variants_generated >= 1


# ============================================================================
# ROUTE TESTS
# ============================================================================


def test_hls_streaming_route_creation(api_client):
    """Test hls_streaming route is registered."""
    response = api_client.get("/openapi.json")
    assert response.status_code == 200

    openapi = response.json()
    assert "/jobs/hls_streaming" in openapi["paths"]


@pytest.mark.requires_ffmpeg
def test_hls_streaming_route_job_submission(api_client, sample_video_path: Path):
    """Test job submission via hls_streaming route."""
    with open(sample_video_path, "rb") as f:
        response = api_client.post(
            "/jobs/hls_streaming",
            files={"file": ("test.mp4", f, "video/mp4")},
            data={"priority": 5},
        )

    assert response.status_code == 200
    data = response.json()

    assert "job_id" in data
    assert data["status"] == "queued"
    assert data["task_type"] == "hls_streaming"


@pytest.mark.requires_ffmpeg
def test_hls_streaming_route_with_custom_variants(api_client, sample_video_path: Path):
    """Test job submission with custom variant configuration."""
    import json

    variants = [
        {"resolution": 720, "bitrate": 3500},
        {"resolution": 480, "bitrate": 1500},
    ]

    with open(sample_video_path, "rb") as f:
        response = api_client.post(
            "/jobs/hls_streaming",
            files={"file": ("test.mp4", f, "video/mp4")},
            data={"variants": json.dumps(variants), "priority": 5},
        )

    assert response.status_code == 200


# ============================================================================
# INTEGRATION TEST (API → Worker)
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_ffmpeg
async def test_hls_streaming_full_job_lifecycle(
    api_client, worker, job_repository, sample_video_path: Path, file_storage
):
    """Test complete flow: API → Repository → Worker → Output."""
    # 1. Submit job via API
    with open(sample_video_path, "rb") as f:
        response = api_client.post(
            "/jobs/hls_streaming",
            files={"file": ("test.mp4", f, "video/mp4")},
            data={"priority": 5},
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
    output = HLSStreamingOutput.model_validate(job.output)
    assert output.variants_generated > 0
    assert output.total_segments > 0


@pytest.mark.integration
@pytest.mark.requires_ffmpeg
async def test_hls_streaming_full_job_lifecycle_with_original(
    api_client, worker, job_repository, sample_video_path: Path, file_storage
):
    """Test complete flow with original quality included."""
    # 1. Submit job via API
    with open(sample_video_path, "rb") as f:
        response = api_client.post(
            "/jobs/hls_streaming",
            files={"file": ("test.mp4", f, "video/mp4")},
            data={"include_original": "true", "priority": 6},
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

    # 4. Validate output includes original
    assert job.output is not None
    output = HLSStreamingOutput.model_validate(job.output)
    assert output.include_original is True
