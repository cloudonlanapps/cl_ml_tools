"""Unit and integration tests for hash computation plugin.

Tests schema validation, MD5/SHA512 algorithms, task execution, routes, and full job lifecycle.
"""

import json
import hashlib
import subprocess
from io import BytesIO
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from PIL import Image

from cl_ml_tools.plugins.hash.algo.generic import sha512hash_generic
from cl_ml_tools.plugins.hash.algo.image import sha512hash_image
from cl_ml_tools.plugins.hash.algo.video import sha512hash_video2
from cl_ml_tools.plugins.hash.algo.md5 import get_md5_hexdigest, validate_md5String
from cl_ml_tools.plugins.hash.schema import HashOutput, HashParams
from cl_ml_tools.plugins.hash.task import HashTask


# ============================================================================
# SCHEMA TESTS
# ============================================================================


def test_hash_params_schema_validation():
    """Test HashParams schema validates correctly."""
    params = HashParams(
        input_path="/path/to/input.jpg",
        output_path="output/hash.json",
        algorithm="sha512",
    )

    assert params.input_path == "/path/to/input.jpg"
    assert params.output_path == "output/hash.json"
    assert params.algorithm == "sha512"


def test_hash_params_defaults():
    """Test HashParams has correct default values."""
    params = HashParams(
        input_path="/path/to/input.jpg",
        output_path="output/hash.json",
    )

    assert params.algorithm == "sha512"


def test_hash_params_algorithm_validation():
    """Test HashParams validates algorithm field."""
    # Valid algorithms
    params1 = HashParams(
        input_path="/path/to/input.jpg",
        output_path="output/hash.json",
        algorithm="sha512",
    )
    assert params1.algorithm == "sha512"

    params2 = HashParams(
        input_path="/path/to/input.jpg",
        output_path="output/hash.json",
        algorithm="md5",
    )
    assert params2.algorithm == "md5"

    # Invalid algorithm should fail validation
    with pytest.raises(ValueError):
        _ = HashParams(
            input_path="/path/to/input.jpg",
            output_path="output/hash.json",
            algorithm="sha256",  # type: ignore
        )


def test_hash_output_schema_validation():
    """Test HashOutput schema validates correctly."""
    output = HashOutput(media_type="image")

    assert output.media_type == "image"


def test_hash_output_optional_media_type():
    """Test HashOutput media_type is optional."""
    output = HashOutput()

    assert output.media_type is None


# ============================================================================
# ALGORITHM TESTS - MD5
# ============================================================================


def test_md5_algo_basic():
    """Test MD5 hash computation with known data."""
    data = b"hello world"
    bytes_io = BytesIO(data)

    hash_value = get_md5_hexdigest(bytes_io)

    # Known MD5 hash of "hello world"
    expected = "5eb63bbbe01eeed093cb22bb8f5acdc3"
    assert hash_value == expected


def test_md5_algo_empty_data():
    """Test MD5 hash computation with empty data."""
    bytes_io = BytesIO(b"")

    hash_value = get_md5_hexdigest(bytes_io)

    # Known MD5 hash of empty string
    expected = "d41d8cd98f00b204e9800998ecf8427e"
    assert hash_value == expected


def test_md5_algo_consistency():
    """Test MD5 hash is consistent across multiple computations."""
    data = b"test data for consistency"
    bytes_io = BytesIO(data)

    hash1 = get_md5_hexdigest(bytes_io)
    hash2 = get_md5_hexdigest(bytes_io)

    assert hash1 == hash2


def test_md5_validate_string():
    """Test MD5 string validation."""
    data = b"validation test"
    bytes_io = BytesIO(data)

    hash_value = get_md5_hexdigest(bytes_io)
    is_valid = validate_md5String(bytes_io, hash_value)

    assert is_valid is True


def test_md5_validate_string_mismatch():
    """Test MD5 string validation with wrong hash."""
    data = b"validation test"
    bytes_io = BytesIO(data)

    is_valid = validate_md5String(bytes_io, "wrong_hash_value")

    assert is_valid is False


# ============================================================================
# ALGORITHM TESTS - SHA512 Generic
# ============================================================================


def test_sha512_generic_algo_basic():
    """Test SHA512 generic hash computation."""
    data = b"test data"
    bytes_io = BytesIO(data)

    hash_value, process_time = sha512hash_generic(bytes_io)

    assert isinstance(hash_value, str)
    assert len(hash_value) == 128  # SHA512 produces 128 hex characters
    assert isinstance(process_time, float)
    assert process_time >= 0


def test_sha512_generic_algo_consistency():
    """Test SHA512 generic hash is consistent."""
    data = b"consistency check"
    bytes_io1 = BytesIO(data)
    bytes_io2 = BytesIO(data)

    hash1, _ = sha512hash_generic(bytes_io1)
    hash2, _ = sha512hash_generic(bytes_io2)

    assert hash1 == hash2


# ============================================================================
# ALGORITHM TESTS - SHA512 Image
# ============================================================================


def test_sha512_image_algo_with_real_image(sample_image_path: Path):
    """Test SHA512 image hash with real image."""
    with open(sample_image_path, "rb") as f:
        bytes_io = BytesIO(f.read())

    hash_value, process_time = sha512hash_image(bytes_io)

    assert isinstance(hash_value, str)
    assert len(hash_value) == 128  # SHA512 produces 128 hex characters
    assert isinstance(process_time, float)
    assert process_time >= 0


def test_sha512_image_algo_consistency(sample_image_path: Path):
    """Test SHA512 image hash is consistent across multiple computations."""
    with open(sample_image_path, "rb") as f:
        data = f.read()

    bytes_io1 = BytesIO(data)
    bytes_io2 = BytesIO(data)

    hash1, _ = sha512hash_image(bytes_io1)
    hash2, _ = sha512hash_image(bytes_io2)

    assert hash1 == hash2


def test_sha512_image_algo_synthetic_image(synthetic_image: Path):
    """Test SHA512 image hash with synthetic image."""
    with open(synthetic_image, "rb") as f:
        bytes_io = BytesIO(f.read())

    hash_value, process_time = sha512hash_image(bytes_io)

    assert isinstance(hash_value, str)
    assert len(hash_value) == 128
    assert process_time >= 0


def test_sha512_image_different_images_different_hashes(
    sample_image_path: Path, synthetic_image: Path
):
    """Test different images produce different SHA512 hashes."""
    with open(sample_image_path, "rb") as f:
        bytes_io1 = BytesIO(f.read())

    with open(synthetic_image, "rb") as f:
        bytes_io2 = BytesIO(f.read())

    hash1, _ = sha512hash_image(bytes_io1)
    hash2, _ = sha512hash_image(bytes_io2)

    assert hash1 != hash2


# ============================================================================
# ALGORITHM TESTS - Different Media Types
# ============================================================================


def test_hash_different_media_types_produce_different_hashes(
    sample_image_path: Path, tmp_path: Path
):
    """Test same data with different media type detection produces different hashes."""
    # Read image data
    with open(sample_image_path, "rb") as f:
        image_data = f.read()

    # Compute image hash
    bytes_io_image = BytesIO(image_data)
    hash_image, _ = sha512hash_image(bytes_io_image)

    # Compute generic hash
    bytes_io_generic = BytesIO(image_data)
    hash_generic, _ = sha512hash_generic(bytes_io_generic)

    # They should be different because image hash uses PIL tobytes()
    assert hash_image != hash_generic


# ============================================================================
# ALGORITHM TESTS - SHA512 Video
# ============================================================================


@pytest.mark.requires_ffmpeg
def test_sha512_video_algo_with_real_video(sample_video_path: Path):
    """Test SHA512 video hash with real video."""
    with open(sample_video_path, "rb") as f:
        video_bytes = f.read()
        bytes_io = BytesIO(video_bytes)

    # Mock ffprobe output because MP4 via stdin might not be seekable/streamable
    # We need to provide a valid CSV row that matches at least one frame in the file
    # For a 30s HD video, there are definitely many I-frames. 
    # Let's mock a simple one-frame hash for testing the loop.
    mock_csv = "0,10,I\n" # offset 0, size 10, type I
    
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = mock_csv.encode("utf-8")
    
    with patch("subprocess.run", return_value=mock_result):
        hash_bytes = sha512hash_video2(bytes_io)

    assert isinstance(hash_bytes, bytes)
    assert len(hash_bytes) == 64
    
    # Verify the hash matches what we expect (SHA512 of first 10 bytes)
    expected_hash = hashlib.sha512(video_bytes[:10]).digest()
    assert hash_bytes == expected_hash


@pytest.mark.requires_ffmpeg
def test_sha512_video_algo_consistency(sample_video_path: Path):
    """Test SHA512 video hash is consistent."""
    with open(sample_video_path, "rb") as f:
        data = f.read()

    bytes_io1 = BytesIO(data)
    bytes_io2 = BytesIO(data)

    mock_csv = "0,10,I\n10,10,P\n20,10,I\n"
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = mock_csv.encode("utf-8")

    with patch("subprocess.run", return_value=mock_result):
        hash1 = sha512hash_video2(bytes_io1)
    
    with patch("subprocess.run", return_value=mock_result):
        hash2 = sha512hash_video2(bytes_io2)

    assert hash1 == hash2
    
    # Expected: hash(data[0:10] + data[20:30])
    expected = hashlib.sha512(data[0:10] + data[20:30]).digest()
    assert hash1 == expected


@pytest.mark.requires_ffmpeg
def test_sha512_video_algo_errors(sample_video_path: Path):
    """Test SHA512 video hash error handling."""
    from cl_ml_tools.plugins.hash.algo.video import UnsupportedMediaType
    bytes_io = BytesIO(b"dummy data")
    
    # 1. Timeout
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)):
        with pytest.raises(UnsupportedMediaType, match="ffprobe command timed out"):
            sha512hash_video2(bytes_io)
            
    # 2. OSError
    with patch("subprocess.run", side_effect=OSError("fail")):
        with pytest.raises(UnsupportedMediaType, match="An error occurred while running ffprobe"):
            sha512hash_video2(bytes_io)
            
    # 3. Non-zero return code
    mock_res = MagicMock(returncode=1, stderr=b"some error")
    with patch("subprocess.run", return_value=mock_res):
        with pytest.raises(UnsupportedMediaType, match="ffprobe error: some error"):
            sha512hash_video2(bytes_io)
            
    # 4. Empty output
    mock_res = MagicMock(returncode=0, stdout=b"  \n")
    with patch("subprocess.run", return_value=mock_res):
        with pytest.raises(UnsupportedMediaType, match="No data returned from ffprobe"):
            sha512hash_video2(bytes_io)

    # 5. Invalid CSV values
    mock_res = MagicMock(returncode=0, stdout=b"abc,10,I\n")
    with patch("subprocess.run", return_value=mock_res):
        with pytest.raises(UnsupportedMediaType, match="CSV data is invalid"):
            sha512hash_video2(bytes_io)

    # 6. Read failure (IOError)
    mock_res = MagicMock(returncode=0, stdout=b"0,10,I\n")
    with patch("subprocess.run", return_value=mock_res):
        with patch.object(bytes_io, "read", side_effect=IOError("read fail")):
            with pytest.raises(UnsupportedMediaType, match="Error processing video data"):
                 sha512hash_video2(bytes_io)

    # 7. Incomplete read
    bytes_io = BytesIO(b"enough data to pass validate_csv") # say 20 bytes
    bytes_io.getbuffer().nbytes # ensuring it's 20
    # Actually just create it with 20 bytes
    bytes_io = BytesIO(b"A" * 20)
    mock_res = MagicMock(returncode=0, stdout=b"0,10,I\n")
    with patch("subprocess.run", return_value=mock_res):
        with patch.object(bytes_io, "read", return_value=b"short"): # returns 5 bytes instead of 10
            with pytest.raises(UnsupportedMediaType, match="Failed to read complete frame"):
                 sha512hash_video2(bytes_io)


# ============================================================================
# TASK TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_hash_task_run_success_md5(sample_image_path: Path, file_storage, tmp_path: Path):
    """Test HashTask execution with MD5 algorithm."""
    job_id = "test-job-123"

    # Set up job storage directory with input file
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_dir = job_dir / "input"
    input_dir.mkdir(exist_ok=True)
    input_file = input_dir / "test.jpg"
    input_file.write_bytes(sample_image_path.read_bytes())

    params = HashParams(
        input_path="input/test.jpg",  # Relative path
        output_path="output/hash.json",
        algorithm="md5",
    )

    task = HashTask()

    # Mock storage with resolve_path
    class MockStorage:
        def resolve_path(self, job_id: str, relative_path: str) -> Path:
            return tmp_path / job_id / relative_path

        def allocate_path(self, job_id: str, relative_path: str) -> str:
            output_path = tmp_path / job_id / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return str(output_path)

    storage = MockStorage()

    output = await task.run(job_id, params, storage)

    assert isinstance(output, HashOutput)
    assert output.media_type == "image"

    # Verify output file
    output_file = tmp_path / job_id / "output" / "hash.json"
    assert output_file.exists()

    with open(output_file) as f:
        result = json.load(f)

    assert "hash_value" in result
    assert result["algorithm"] == "md5"
    assert result["media_type"] == "image"
    assert "process_time" in result


@pytest.mark.asyncio
async def test_hash_task_run_success_sha512(sample_image_path: Path, tmp_path: Path):
    """Test HashTask execution with SHA512 algorithm."""
    job_id = "test-job-456"

    # Set up job storage directory with input file
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_dir = job_dir / "input"
    input_dir.mkdir(exist_ok=True)
    input_file = input_dir / "test.jpg"
    input_file.write_bytes(sample_image_path.read_bytes())

    params = HashParams(
        input_path="input/test.jpg",  # Relative path
        output_path="output/hash.json",
        algorithm="sha512",
    )

    task = HashTask()

    # Mock storage with resolve_path
    class MockStorage:
        def resolve_path(self, job_id: str, relative_path: str) -> Path:
            return tmp_path / job_id / relative_path

        def allocate_path(self, job_id: str, relative_path: str) -> str:
            output_path = tmp_path / job_id / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return str(output_path)

    storage = MockStorage()

    output = await task.run(job_id, params, storage)

    assert isinstance(output, HashOutput)
    assert output.media_type == "image"

    # Verify output file
    output_file = tmp_path / job_id / "output" / "hash.json"
    assert output_file.exists()

    with open(output_file) as f:
        result = json.load(f)

    assert "hash_value" in result
    assert result["algorithm"] == "sha512_image"
    assert result["media_type"] == "image"


@pytest.mark.asyncio
async def test_hash_task_run_file_not_found(tmp_path: Path):
    """Test HashTask raises FileNotFoundError for missing input."""
    params = HashParams(
        input_path="input/nonexistent.jpg",  # Relative path
        output_path="output/hash.json",
    )

    task = HashTask()
    job_id = "test-job-789"

    # Mock storage with resolve_path
    class MockStorage:
        def resolve_path(self, job_id: str, relative_path: str) -> Path:
            return tmp_path / job_id / relative_path

        def allocate_path(self, job_id: str, relative_path: str) -> str:
            return str(tmp_path / job_id / relative_path)

    storage = MockStorage()

    with pytest.raises(FileNotFoundError):
        await task.run(job_id, params, storage)


@pytest.mark.asyncio
async def test_hash_task_progress_callback(sample_image_path: Path, tmp_path: Path):
    """Test HashTask calls progress callback."""
    job_id = "test-job-progress"

    # Set up job storage directory with input file
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_dir = job_dir / "input"
    input_dir.mkdir(exist_ok=True)
    input_file = input_dir / "test.jpg"
    input_file.write_bytes(sample_image_path.read_bytes())

    params = HashParams(
        input_path="input/test.jpg",  # Relative path
        output_path="output/hash.json",
    )

    task = HashTask()

    # Mock storage with resolve_path
    class MockStorage:
        def resolve_path(self, job_id: str, relative_path: str) -> Path:
            return tmp_path / job_id / relative_path

        def allocate_path(self, job_id: str, relative_path: str) -> str:
            output_path = tmp_path / job_id / relative_path
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


def test_hash_route_creation(api_client):
    """Test hash route is registered."""
    response = api_client.get("/openapi.json")
    assert response.status_code == 200

    openapi = response.json()
    assert "/jobs/hash" in openapi["paths"]


def test_hash_route_job_submission(api_client, sample_image_path: Path):
    """Test job submission via hash route."""
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/hash",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"algorithm": "md5", "priority": 5},
        )

    assert response.status_code == 200
    data = response.json()

    assert "job_id" in data
    assert data["status"] == "queued"
    assert data["task_type"] == "hash"


def test_hash_route_job_submission_sha512(api_client, sample_image_path: Path):
    """Test job submission with SHA512 algorithm."""
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/hash",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"algorithm": "sha512", "priority": 7},
        )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "queued"
    assert data["task_type"] == "hash"


def test_hash_route_default_algorithm(api_client, sample_image_path: Path):
    """Test hash route uses default algorithm when not specified."""
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/hash",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"priority": 5},
        )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "queued"


# ============================================================================
# INTEGRATION TEST (API → Worker)
# ============================================================================


@pytest.mark.integration
async def test_hash_full_job_lifecycle(
    api_client, worker, job_repository, sample_image_path: Path, file_storage
):
    """Test complete flow: API → Repository → Worker → Output."""
    # 1. Submit job via API
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/hash",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"algorithm": "md5", "priority": 5},
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
    output = HashOutput.model_validate(job.output)
    assert output.media_type == "image"


@pytest.mark.integration
async def test_hash_full_job_lifecycle_sha512(
    api_client, worker, job_repository, sample_image_path: Path, file_storage
):
    """Test complete flow with SHA512 algorithm."""
    # 1. Submit job via API
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/hash",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"algorithm": "sha512", "priority": 8},
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
    output = HashOutput.model_validate(job.output)
    assert output.media_type == "image"
