"""Comprehensive test suite for hash plugin."""

import hashlib
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import pytest
from PIL import Image
from pydantic import ValidationError

from cl_ml_tools.common.schemas import Job
from cl_ml_tools.plugins.hash import HashParams, HashTask
from cl_ml_tools.plugins.hash.algo.generic import sha512hash_generic
from cl_ml_tools.plugins.hash.algo.image import sha512hash_image
from cl_ml_tools.plugins.hash.algo.md5 import get_md5_hexdigest
from cl_ml_tools.utils.random_media_generator import RandomMediaGenerator

# Helper type for params dict
ParamsDict = dict[str, list[str] | Literal["sha512", "md5"]]


# Helper class for mock progress callback
class MockProgressCallback:
    """Mock progress callback for testing."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, progress: int) -> None:
        self.calls.append(progress)


# ─────────────────────────────────────────────────────────────
# 1. Schema Tests
# ─────────────────────────────────────────────────────────────


class TestHashParams:
    """Test HashParams schema validation."""

    def test_default_algorithm(self):
        """Test algorithm defaults to sha512."""
        params = HashParams(input_paths=["/test/input.txt"], output_paths=[])
        assert params.algorithm == "sha512"

    def test_sha512_algorithm(self):
        """Test sha512 algorithm is accepted."""
        params = HashParams(
            input_paths=["/test/input.txt"],
            output_paths=[],
            algorithm="sha512",
        )
        assert params.algorithm == "sha512"

    def test_md5_algorithm(self):
        """Test md5 algorithm is accepted."""
        params = HashParams(input_paths=["/test/input.txt"], output_paths=[], algorithm="md5")
        assert params.algorithm == "md5"

    def test_invalid_algorithm(self):
        """Test invalid algorithm is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            # Use dict unpacking to bypass type checking for invalid value
            invalid_params = {
                "input_paths": ["/test/input.txt"],
                "output_paths": [],
                "algorithm": "invalid",
            }
            _ = HashParams(**invalid_params)
        assert "algorithm" in str(exc_info.value).lower()

    def test_empty_output_paths(self):
        """Test output_paths can be empty (compute-only)."""
        params = HashParams(input_paths=["/test/input.txt"], output_paths=[])
        assert params.output_paths == []

    def test_inherited_validations(self):
        """Test BaseJobParams validations still apply."""
        # Test duplicate output paths (should fail if output_paths provided)
        with pytest.raises(ValidationError) as exc_info:
            _ = HashParams(
                input_paths=["/test/input1.txt", "/test/input2.txt"],
                output_paths=["/test/same.txt", "/test/same.txt"],
                algorithm="sha512",
            )
        assert "unique" in str(exc_info.value).lower()


# ─────────────────────────────────────────────────────────────
# 2. Generic Algorithm Tests
# ─────────────────────────────────────────────────────────────


class TestSHA512Generic:
    """Test sha512hash_generic function."""

    def test_empty_file(self):
        """Test hash of empty BytesIO."""
        bytes_io = BytesIO(b"")
        hash_value, process_time = sha512hash_generic(bytes_io)

        # Verify it's a valid SHA-512 hash
        assert len(hash_value) == 128  # SHA-512 produces 128 hex characters
        assert isinstance(process_time, float)
        assert process_time >= 0

        # Verify it matches expected empty hash
        expected = hashlib.sha512(b"").hexdigest()
        assert hash_value == expected

    def test_small_text_file(self):
        """Test hash of small text content."""
        content = b"Hello, World!"
        bytes_io = BytesIO(content)
        hash_value, process_time = sha512hash_generic(bytes_io)

        assert len(hash_value) == 128
        assert isinstance(process_time, float)

        # Verify against expected hash
        expected = hashlib.sha512(content).hexdigest()
        assert hash_value == expected

    def test_large_file(self):
        """Test chunked reading with large file (>4096 bytes)."""
        # Create 10KB of data
        content = b"a" * (10 * 1024)
        bytes_io = BytesIO(content)
        hash_value, process_time = sha512hash_generic(bytes_io)

        assert len(hash_value) == 128
        assert isinstance(process_time, float)

        # Verify chunked reading produces same result
        expected = hashlib.sha512(content).hexdigest()
        assert hash_value == expected

    def test_deterministic_hash(self):
        """Test same input produces same hash."""
        content = b"Test content"
        bytes_io1 = BytesIO(content)
        bytes_io2 = BytesIO(content)

        hash1, _ = sha512hash_generic(bytes_io1)
        hash2, _ = sha512hash_generic(bytes_io2)

        assert hash1 == hash2

    def test_different_inputs(self):
        """Test different inputs produce different hashes."""
        bytes_io1 = BytesIO(b"Content 1")
        bytes_io2 = BytesIO(b"Content 2")

        hash1, _ = sha512hash_generic(bytes_io1)
        hash2, _ = sha512hash_generic(bytes_io2)

        assert hash1 != hash2

    def test_return_format(self):
        """Test returns tuple (hash_str, process_time)."""
        bytes_io = BytesIO(b"test")
        result = sha512hash_generic(bytes_io)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], float)

    def test_hash_length(self):
        """Test SHA-512 hash is 128 hex characters."""
        bytes_io = BytesIO(b"test data")
        hash_value, _ = sha512hash_generic(bytes_io)

        assert len(hash_value) == 128
        # Verify all characters are hex digits
        assert all(c in "0123456789abcdef" for c in hash_value)


# ─────────────────────────────────────────────────────────────
# 3. Image Algorithm Tests
# ─────────────────────────────────────────────────────────────


class TestImageHash:
    """Test sha512hash_image function."""

    @pytest.fixture
    def sample_image(self) -> BytesIO:
        """Create test image in memory."""
        # Create a simple 10x10 red image
        img = Image.new("RGB", (10, 10), color="red")
        bytes_io = BytesIO()
        img.save(bytes_io, format="PNG")
        _ = bytes_io.seek(0)
        return bytes_io

    def test_image_hash(self, sample_image: BytesIO) -> None:
        """Test hashing a valid image."""
        hash_value, process_time = sha512hash_image(sample_image)

        assert len(hash_value) == 128
        assert isinstance(process_time, float)
        assert process_time >= 0

    def test_image_hash_deterministic(self, sample_image: BytesIO) -> None:
        """Test same image produces same hash."""
        # Create two identical images
        img = Image.new("RGB", (10, 10), color="blue")

        bytes_io1 = BytesIO()
        img.save(bytes_io1, format="PNG")
        _ = bytes_io1.seek(0)

        bytes_io2 = BytesIO()
        img.save(bytes_io2, format="PNG")
        _ = bytes_io2.seek(0)

        hash1, _ = sha512hash_image(bytes_io1)
        hash2, _ = sha512hash_image(bytes_io2)

        # Note: Same pixel data should produce same hash
        # even if PNG encoding differs slightly
        assert hash1 == hash2

    def test_return_format(self, sample_image: BytesIO) -> None:
        """Test returns (hash_str, time)."""
        result = sha512hash_image(sample_image)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], float)

    def test_invalid_image(self):
        """Test with corrupt image data."""
        bytes_io = BytesIO(b"not an image")

        with pytest.raises(Exception):
            # Should raise PIL exception for invalid image
            _ = sha512hash_image(bytes_io)


# ─────────────────────────────────────────────────────────────
# 4. Video Algorithm Tests
# ─────────────────────────────────────────────────────────────


class TestVideoHash:
    """Test sha512hash_video2 function."""

    def test_video_hash_with_random_video(self):
        """Test video hashing with generated video file."""
        from cl_ml_tools.plugins.hash.algo.video import sha512hash_video2

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Generate a test video using random_media_generator
            gen = RandomMediaGenerator(
                out_dir=tmp_dir,
                media_list=[
                    {
                        "MIMEType": "video/mp4",
                        "width": 320,
                        "height": 240,
                        "fileName": "test_video",
                        "scenes": [
                            {
                                "background_color": [100, 150, 200],
                                "num_shapes": 2,
                                "duration_seconds": 1,
                            }
                        ],
                        "fps": 10,
                    }
                ],
            )

            # Generate the video
            video_media = gen.media_list[0]
            video_media.generate()

            # Read the generated video into BytesIO
            video_path = Path(video_media.filepath)
            assert video_path.exists(), "Generated video file should exist"

            video_bytes = video_path.read_bytes()
            bytes_io = BytesIO(video_bytes)

            # Compute hash
            hash_value = sha512hash_video2(bytes_io)

            # Verify result is bytes and has correct length (SHA-512 = 64 bytes)
            assert isinstance(hash_value, bytes)
            assert len(hash_value) == 64

    def test_return_format_is_bytes(self):
        """Test that function signature returns bytes."""
        import inspect

        from cl_ml_tools.plugins.hash.algo.video import sha512hash_video2

        sig = inspect.signature(sha512hash_video2)
        # Function should return bytes (not str)
        assert sig.return_annotation is bytes


# ─────────────────────────────────────────────────────────────
# 5. MD5 Algorithm Tests
# ─────────────────────────────────────────────────────────────


class TestMD5Hash:
    """Test get_md5_hexdigest function."""

    def test_md5_empty(self):
        """Test MD5 of empty BytesIO."""
        bytes_io = BytesIO(b"")
        hash_value = get_md5_hexdigest(bytes_io)

        assert len(hash_value) == 32  # MD5 produces 32 hex characters
        expected = hashlib.md5(b"").hexdigest()
        assert hash_value == expected

    def test_md5_text(self):
        """Test MD5 of text content."""
        content = b"Hello, MD5!"
        bytes_io = BytesIO(content)
        hash_value = get_md5_hexdigest(bytes_io)

        assert len(hash_value) == 32
        expected = hashlib.md5(content).hexdigest()
        assert hash_value == expected

    def test_md5_deterministic(self):
        """Test same input produces same hash."""
        content = b"Test content"
        bytes_io1 = BytesIO(content)
        bytes_io2 = BytesIO(content)

        hash1 = get_md5_hexdigest(bytes_io1)
        hash2 = get_md5_hexdigest(bytes_io2)

        assert hash1 == hash2

    def test_md5_length(self):
        """Test MD5 hash is 32 hex characters."""
        bytes_io = BytesIO(b"test data")
        hash_value = get_md5_hexdigest(bytes_io)

        assert len(hash_value) == 32
        assert all(c in "0123456789abcdef" for c in hash_value)


# ─────────────────────────────────────────────────────────────
# 6. Task Execution Tests
# ─────────────────────────────────────────────────────────────


class TestHashTask:
    """Test HashTask ComputeModule implementation."""

    @pytest.fixture
    def hash_task(self) -> HashTask:
        """Create HashTask instance."""
        return HashTask()

    @pytest.fixture
    def temp_text_file(self, tmp_path: Any) -> str:
        """Create temporary text file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello, World!")
        return str(file_path)

    @pytest.fixture
    def temp_image_file(self, tmp_path: Any) -> str:
        """Create temporary image file."""
        file_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(file_path)
        return str(file_path)

    @pytest.fixture
    def mock_progress_callback(self) -> MockProgressCallback:
        """Create mock progress callback that tracks calls."""
        return MockProgressCallback()

    def test_task_type(self, hash_task):
        """Test task_type returns 'hash'."""
        assert hash_task.task_type == "hash"

    def test_get_schema(self, hash_task):
        """Test get_schema returns HashParams."""
        schema = hash_task.get_schema()
        assert schema == HashParams

    async def test_execute_text_file_sha512(self, hash_task: HashTask, temp_text_file: str) -> None:
        """Test execute with text file and sha512 algorithm."""
        params = HashParams(
            input_paths=[temp_text_file],
            output_paths=[],
            algorithm="sha512",
        )
        job = Job(
            job_id="test-job-1",
            task_type="hash",
            params={
                "input_paths": [temp_text_file],
                "output_paths": [],
                "algorithm": "sha512",
            },
        )

        result = await hash_task.execute(job, params, None)

        assert result["status"] == "ok"
        assert "task_output" in result
        task_output = result["task_output"]
        assert task_output["total_files"] == 1
        assert len(task_output["files"]) == 1

        file_result = task_output["files"][0]
        assert file_result["file_path"] == temp_text_file
        assert file_result["media_type"] == "text"
        assert file_result["algorithm_used"] == "sha512_generic"
        assert len(file_result["hash_value"]) == 128
        assert isinstance(file_result["process_time"], float)

    async def test_execute_text_file_md5(self, hash_task: HashTask, temp_text_file: str) -> None:
        """Test execute with text file and md5 algorithm."""
        params_dict = {
            "input_paths": [temp_text_file],
            "output_paths": [],
            "algorithm": "md5",
        }
        job = Job(
            job_id="test-job-2",
            task_type="hash",
            params=params_dict,
        )
        params = HashParams(**params_dict)

        result = await hash_task.execute(job, params, None)

        assert result["status"] == "ok"
        assert result["task_output"] is not None
        file_result = result["task_output"]["files"][0]
        assert file_result["algorithm_used"] == "md5"
        assert len(str(file_result["hash_value"])) == 32

    async def test_execute_image_file(self, hash_task: HashTask, temp_image_file: str) -> None:
        """Test execute with image file."""
        params_dict = {
            "input_paths": [temp_image_file],
            "output_paths": [],
            "algorithm": "sha512",
        }
        job = Job(
            job_id="test-job-3",
            task_type="hash",
            params=params_dict,
        )
        params = HashParams(**params_dict)

        result = await hash_task.execute(job, params, None)

        assert result["status"] == "ok"
        assert result["task_output"] is not None
        file_result = result["task_output"]["files"][0]
        assert file_result["media_type"] == "image"
        assert file_result["algorithm_used"] == "sha512_image"
        assert len(str(file_result["hash_value"])) == 128

    async def test_execute_with_progress_callback(
        self,
        hash_task: HashTask,
        temp_text_file: str,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test progress callback is invoked."""
        params_dict = {
            "input_paths": [temp_text_file],
            "output_paths": [],
            "algorithm": "sha512",
        }
        job = Job(
            job_id="test-job-4",
            task_type="hash",
            params=params_dict,
        )
        params = HashParams(**params_dict)

        result = await hash_task.execute(job, params, mock_progress_callback)

        assert result["status"] == "ok"
        # Progress callback should be called once (100% for single file)
        assert len(mock_progress_callback.calls) == 1
        assert mock_progress_callback.calls[0] == 100

    async def test_execute_file_not_found(self, hash_task: HashTask) -> None:
        """Test execute with non-existent file."""
        params_dict = {
            "input_paths": ["/nonexistent/file.txt"],
            "output_paths": [],
            "algorithm": "sha512",
        }
        job = Job(
            job_id="test-job-5",
            task_type="hash",
            params=params_dict,
        )
        params = HashParams(**params_dict)

        result = await hash_task.execute(job, params, None)

        assert result["status"] == "error"
        assert "error" in result
        assert "not found" in str(result["error"]).lower()

    async def test_execute_multiple_files(self, hash_task: HashTask, tmp_path: Any) -> None:
        """Test batch processing of multiple files."""
        # Create multiple test files
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        params_dict = {
            "input_paths": [str(file1), str(file2)],
            "output_paths": [],
            "algorithm": "sha512",
        }
        job = Job(
            job_id="test-job-6",
            task_type="hash",
            params=params_dict,
        )
        params = HashParams(**params_dict)

        result = await hash_task.execute(job, params, None)

        assert result["status"] == "ok"
        assert result["task_output"] is not None
        assert result["task_output"]["total_files"] == 2
        assert len(result["task_output"]["files"]) == 2

        # Verify different content produces different hashes
        hash1 = result["task_output"]["files"][0]["hash_value"]
        hash2 = result["task_output"]["files"][1]["hash_value"]
        assert hash1 != hash2

    async def test_output_structure(self, hash_task: HashTask, temp_text_file: str) -> None:
        """Test task_output has correct structure."""
        params_dict = {
            "input_paths": [temp_text_file],
            "output_paths": [],
            "algorithm": "sha512",
        }
        job = Job(
            job_id="test-job-7",
            task_type="hash",
            params=params_dict,
        )
        params = HashParams(**params_dict)

        result = await hash_task.execute(job, params, None)

        # Verify top-level structure
        assert "status" in result
        assert "task_output" in result

        # Verify task_output structure
        task_output = result["task_output"]
        assert task_output is not None
        assert "files" in task_output
        assert "total_files" in task_output

        # Verify file result structure
        file_result = task_output["files"][0]
        assert "file_path" in file_result
        assert "media_type" in file_result
        assert "hash_value" in file_result
        assert "algorithm_used" in file_result
        assert "process_time" in file_result

    async def test_media_type_detection(self, hash_task: HashTask, temp_text_file: str) -> None:
        """Test media type is correctly detected and included."""
        params_dict = {
            "input_paths": [temp_text_file],
            "output_paths": [],
            "algorithm": "sha512",
        }
        job = Job(
            job_id="test-job-8",
            task_type="hash",
            params=params_dict,
        )
        params = HashParams(**params_dict)

        result = await hash_task.execute(job, params, None)

        assert result["task_output"] is not None
        file_result = result["task_output"]["files"][0]
        # Text file should be detected as "text" media type
        assert file_result["media_type"] in ["text", "file"]  # Could be either


# ─────────────────────────────────────────────────────────────
# 7. Integration Tests
# ─────────────────────────────────────────────────────────────


class TestHashPluginIntegration:
    """Test hash plugin end-to-end integration."""

    async def test_text_file_routing(self, tmp_path: Any) -> None:
        """Test text file routes to sha512_generic."""
        file_path = tmp_path / "document.txt"
        file_path.write_text("Sample document content")

        task = HashTask()
        params_dict = {
            "input_paths": [str(file_path)],
            "output_paths": [],
            "algorithm": "sha512",
        }
        job = Job(
            job_id="integration-1",
            task_type="hash",
            params=params_dict,
        )
        params = HashParams(**params_dict)

        result = await task.execute(job, params, None)

        assert result["status"] == "ok"
        assert result["task_output"] is not None
        file_result = result["task_output"]["files"][0]
        assert file_result["algorithm_used"] == "sha512_generic"

    async def test_image_file_routing(self, tmp_path: Any) -> None:
        """Test image file routes to sha512_image."""
        file_path = tmp_path / "image.png"
        img = Image.new("RGB", (50, 50), color="green")
        img.save(file_path)

        task = HashTask()
        params_dict = {
            "input_paths": [str(file_path)],
            "output_paths": [],
            "algorithm": "sha512",
        }
        job = Job(
            job_id="integration-2",
            task_type="hash",
            params=params_dict,
        )
        params = HashParams(**params_dict)

        result = await task.execute(job, params, None)

        assert result["status"] == "ok"
        assert result["task_output"] is not None
        file_result = result["task_output"]["files"][0]
        assert file_result["media_type"] == "image"
        assert file_result["algorithm_used"] == "sha512_image"

    async def test_algorithm_selection(self, tmp_path: Any) -> None:
        """Test algorithm parameter changes routing."""
        file_path = tmp_path / "data.txt"
        file_path.write_text("Test data")

        task = HashTask()

        # Test SHA-512
        params_dict_sha = {
            "input_paths": [str(file_path)],
            "output_paths": [],
            "algorithm": "sha512",
        }
        job_sha = Job(
            job_id="integration-3a",
            task_type="hash",
            params=params_dict_sha,
        )
        params_sha = HashParams(**params_dict_sha)
        result_sha = await task.execute(job_sha, params_sha, None)

        # Test MD5
        params_dict_md5 = {
            "input_paths": [str(file_path)],
            "output_paths": [],
            "algorithm": "md5",
        }
        job_md5 = Job(
            job_id="integration-3b",
            task_type="hash",
            params=params_dict_md5,
        )
        params_md5 = HashParams(**params_dict_md5)
        result_md5 = await task.execute(job_md5, params_md5, None)

        # Both should succeed
        assert result_sha["status"] == "ok"
        assert result_md5["status"] == "ok"

        # SHA-512 should produce 128-char hash
        assert result_sha["task_output"] is not None
        sha_hash = result_sha["task_output"]["files"][0]["hash_value"]
        assert len(str(sha_hash)) == 128

        # MD5 should produce 32-char hash
        assert result_md5["task_output"] is not None
        md5_hash = result_md5["task_output"]["files"][0]["hash_value"]
        assert len(str(md5_hash)) == 32

    async def test_error_handling(self, tmp_path: Any) -> None:
        """Test error responses have correct structure."""
        task = HashTask()
        params_dict = {
            "input_paths": ["/this/file/does/not/exist.txt"],
            "output_paths": [],
            "algorithm": "sha512",
        }
        job = Job(
            job_id="integration-4",
            task_type="hash",
            params=params_dict,
        )
        params = HashParams(**params_dict)

        result = await task.execute(job, params, None)

        # Should return error status
        assert result["status"] == "error"
        assert "error" in result
        error_val = result.get("error")
        assert error_val is not None
        assert isinstance(error_val, str)
        assert len(error_val) > 0
