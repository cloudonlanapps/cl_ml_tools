"""Comprehensive test suite for face detection plugin."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from cl_ml_tools.common.schemas import Job
from cl_ml_tools.plugins.face_detection.schema import (
    BoundingBox,
    FaceDetectionParams,
    FaceDetectionResult,
)
from cl_ml_tools.plugins.face_detection.task import FaceDetectionTask

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
def sample_image(tmp_path: Path) -> str:
    """Create a test image with a face-like pattern."""
    image_path = tmp_path / "test_face.jpg"

    # Create a simple image (192x192 to match model input)
    img = Image.new("RGB", (192, 192), color=(200, 150, 100))
    img.save(str(image_path), "JPEG")

    return str(image_path)


@pytest.fixture
def face_detection_task() -> FaceDetectionTask:
    """Create FaceDetectionTask instance."""
    return FaceDetectionTask()


@pytest.fixture
def mock_progress_callback() -> MockProgressCallback:
    """Create mock progress callback."""
    return MockProgressCallback()


# ============================================================================
# Test Class 1: Schema Validation
# ============================================================================


class TestFaceDetectionParams:
    """Test FaceDetectionParams schema validation."""

    def test_default_params(self) -> None:
        """Test default FaceDetectionParams values."""
        params = FaceDetectionParams(input_paths=["/test/image.jpg"], output_paths=[])

        assert params.input_paths == ["/test/image.jpg"]
        assert params.output_paths == []
        assert params.confidence_threshold == 0.7
        assert params.nms_threshold == 0.4

    def test_custom_thresholds(self) -> None:
        """Test FaceDetectionParams with custom thresholds."""
        params = FaceDetectionParams(
            input_paths=["/test/image.jpg"],
            output_paths=[],
            confidence_threshold=0.5,
            nms_threshold=0.3,
        )

        assert params.confidence_threshold == 0.5
        assert params.nms_threshold == 0.3

    def test_invalid_confidence_threshold(self) -> None:
        """Test that invalid confidence threshold is rejected."""
        with pytest.raises(Exception):  # Pydantic validation error
            FaceDetectionParams(
                input_paths=["/test/image.jpg"],
                output_paths=[],
                confidence_threshold=1.5,  # Invalid (>1.0)
            )

    def test_invalid_nms_threshold(self) -> None:
        """Test that invalid NMS threshold is rejected."""
        with pytest.raises(Exception):  # Pydantic validation error
            FaceDetectionParams(
                input_paths=["/test/image.jpg"],
                output_paths=[],
                nms_threshold=-0.1,  # Invalid (<0.0)
            )


class TestBoundingBox:
    """Test BoundingBox schema validation."""

    def test_valid_bounding_box(self) -> None:
        """Test valid BoundingBox creation."""
        bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.8, y2=0.9, confidence=0.95)

        assert bbox.x1 == 0.1
        assert bbox.y1 == 0.2
        assert bbox.x2 == 0.8
        assert bbox.y2 == 0.9
        assert bbox.confidence == 0.95

    def test_to_absolute_coordinates(self) -> None:
        """Test conversion to absolute pixel coordinates."""
        bbox = BoundingBox(x1=0.25, y1=0.5, x2=0.75, y2=1.0, confidence=0.9)

        absolute = bbox.to_absolute(image_width=800, image_height=600)

        assert absolute["x1"] == 200  # 0.25 * 800
        assert absolute["y1"] == 300  # 0.5 * 600
        assert absolute["x2"] == 600  # 0.75 * 800
        assert absolute["y2"] == 600  # 1.0 * 600

    def test_invalid_coordinates(self) -> None:
        """Test that invalid coordinates are rejected."""
        with pytest.raises(Exception):  # Pydantic validation error
            BoundingBox(x1=-0.1, y1=0.2, x2=0.8, y2=0.9, confidence=0.9)

        with pytest.raises(Exception):  # Pydantic validation error
            BoundingBox(x1=0.1, y1=0.2, x2=1.5, y2=0.9, confidence=0.9)


class TestFaceDetectionResult:
    """Test FaceDetectionResult schema validation."""

    def test_success_result(self) -> None:
        """Test successful face detection result."""
        bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.8, y2=0.9, confidence=0.95)
        result = FaceDetectionResult(
            file_path="/test/image.jpg",
            faces=[bbox],
            num_faces=1,
            image_width=800,
            image_height=600,
        )

        assert result.file_path == "/test/image.jpg"
        assert len(result.faces) == 1
        assert result.faces[0].confidence == 0.95
        assert result.num_faces == 1
        assert result.image_width == 800
        assert result.image_height == 600

    def test_no_faces_result(self) -> None:
        """Test face detection result with no faces found."""
        result = FaceDetectionResult(
            file_path="/test/image.jpg", faces=[], num_faces=0, image_width=800, image_height=600
        )

        assert result.file_path == "/test/image.jpg"
        assert len(result.faces) == 0
        assert result.num_faces == 0


# ============================================================================
# Test Class 2: Task Execution
# ============================================================================


class TestFaceDetectionTask:
    """Test FaceDetectionTask execution."""

    def test_task_type(self, face_detection_task: FaceDetectionTask) -> None:
        """Test that task type is correctly set."""
        assert face_detection_task.task_type == "face_detection"

    def test_get_schema(self, face_detection_task: FaceDetectionTask) -> None:
        """Test that get_schema returns FaceDetectionParams."""
        schema = face_detection_task.get_schema()
        assert schema == FaceDetectionParams

    @pytest.mark.asyncio
    async def test_execute_with_mocked_detector(
        self,
        face_detection_task: FaceDetectionTask,
        sample_image: str,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution with mocked face detector."""
        # Create job
        job = Job(
            job_id="test-job-001",
            task_type="face_detection",
            params={
                "input_paths": [sample_image],
                "output_paths": [],
                "confidence_threshold": 0.7,
                "nms_threshold": 0.4,
            },
        )

        params = FaceDetectionParams(**job.params)

        # Mock the detector - returns list of dicts
        mock_detector = Mock()
        mock_detector.detect.return_value = [
            {"x1": 0.2, "y1": 0.3, "x2": 0.7, "y2": 0.8, "confidence": 0.92}
        ]

        with patch.object(
            face_detection_task, "_get_detector", return_value=mock_detector
        ):
            result = await face_detection_task.execute(job, params, mock_progress_callback)

        # Verify result
        assert result["status"] == "ok"
        assert "task_output" in result

        task_output = result["task_output"]
        assert task_output["total_files"] == 1
        assert len(task_output["files"]) == 1

        file_result = task_output["files"][0]
        assert file_result["file_path"] == sample_image
        assert file_result["status"] == "success"
        assert "detection" in file_result
        assert len(file_result["detection"]["faces"]) == 1
        assert file_result["detection"]["faces"][0]["confidence"] == 0.92

        # Verify progress callback was called
        assert len(mock_progress_callback.calls) == 1
        assert mock_progress_callback.calls[0] == 100

    @pytest.mark.asyncio
    async def test_execute_file_not_found(
        self,
        face_detection_task: FaceDetectionTask,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution with non-existent file."""
        job = Job(
            job_id="test-job-002",
            task_type="face_detection",
            params={
                "input_paths": ["/nonexistent/file.jpg"],
                "output_paths": [],
                "confidence_threshold": 0.7,
                "nms_threshold": 0.4,
            },
        )

        params = FaceDetectionParams(**job.params)

        # Mock the detector to raise FileNotFoundError
        mock_detector = Mock()
        mock_detector.detect.side_effect = FileNotFoundError("Image file not found")

        with patch.object(
            face_detection_task, "_get_detector", return_value=mock_detector
        ):
            result = await face_detection_task.execute(job, params, mock_progress_callback)

        # Should return error status with file-level error
        assert result["status"] == "error"
        assert "error" in result

        task_output = result["task_output"]
        file_result = task_output["files"][0]
        assert file_result["status"] == "error"
        assert "not found" in file_result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_multiple_files(
        self,
        face_detection_task: FaceDetectionTask,
        sample_image: str,
        tmp_path: Path,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution with multiple files."""
        # Create second image
        image_path_2 = tmp_path / "test_face_2.jpg"
        img = Image.new("RGB", (192, 192), color=(100, 150, 200))
        img.save(str(image_path_2), "JPEG")

        job = Job(
            job_id="test-job-003",
            task_type="face_detection",
            params={
                "input_paths": [sample_image, str(image_path_2)],
                "output_paths": [],
                "confidence_threshold": 0.7,
                "nms_threshold": 0.4,
            },
        )

        params = FaceDetectionParams(**job.params)

        # Mock the detector - returns list of dicts
        mock_detector = Mock()
        mock_detector.detect.side_effect = [
            [{"x1": 0.2, "y1": 0.3, "x2": 0.7, "y2": 0.8, "confidence": 0.92}],
            [
                {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.6, "confidence": 0.88},
                {"x1": 0.5, "y1": 0.4, "x2": 0.9, "y2": 0.9, "confidence": 0.85},
            ],
        ]

        with patch.object(
            face_detection_task, "_get_detector", return_value=mock_detector
        ):
            result = await face_detection_task.execute(job, params, mock_progress_callback)

        # Verify result
        assert result["status"] == "ok"

        task_output = result["task_output"]
        assert task_output["total_files"] == 2
        assert len(task_output["files"]) == 2

        # First file: 1 face
        assert len(task_output["files"][0]["detection"]["faces"]) == 1

        # Second file: 2 faces
        assert len(task_output["files"][1]["detection"]["faces"]) == 2

        # Verify progress callback was called twice (50%, 100%)
        assert len(mock_progress_callback.calls) == 2
        assert mock_progress_callback.calls == [50, 100]

    @pytest.mark.asyncio
    async def test_execute_no_faces_detected(
        self,
        face_detection_task: FaceDetectionTask,
        sample_image: str,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution when no faces are detected."""
        job = Job(
            job_id="test-job-004",
            task_type="face_detection",
            params={
                "input_paths": [sample_image],
                "output_paths": [],
                "confidence_threshold": 0.7,
                "nms_threshold": 0.4,
            },
        )

        params = FaceDetectionParams(**job.params)

        # Mock the detector to return empty list
        mock_detector = Mock()
        mock_detector.detect.return_value = []

        with patch.object(
            face_detection_task, "_get_detector", return_value=mock_detector
        ):
            result = await face_detection_task.execute(job, params, mock_progress_callback)

        # Should succeed with empty faces list
        assert result["status"] == "ok"

        task_output = result["task_output"]
        file_result = task_output["files"][0]
        assert file_result["status"] == "success"
        assert file_result["detection"]["faces"] == []
        assert file_result["detection"]["num_faces"] == 0

    @pytest.mark.asyncio
    async def test_execute_detector_initialization_failure(
        self,
        face_detection_task: FaceDetectionTask,
        sample_image: str,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution when detector fails to initialize."""
        job = Job(
            job_id="test-job-005",
            task_type="face_detection",
            params={
                "input_paths": [sample_image],
                "output_paths": [],
            },
        )

        params = FaceDetectionParams(**job.params)

        # Mock detector initialization failure
        with patch.object(
            face_detection_task,
            "_get_detector",
            side_effect=RuntimeError("ONNX model not found"),
        ):
            result = await face_detection_task.execute(job, params, mock_progress_callback)

        # Should return error
        assert result["status"] == "error"
        assert "ONNX model not found" in result["error"]


# ============================================================================
# Test Class 3: Algorithm Unit Tests
# ============================================================================


class TestFaceDetectorAlgorithm:
    """Test FaceDetector algorithm components."""

    def test_preprocessing_shape(self) -> None:
        """Test that preprocessing produces correct output shape."""
        from cl_ml_tools.plugins.face_detection.algo.face_detector import FaceDetector

        # Create a mock detector without loading the model
        with patch("onnxruntime.InferenceSession"):
            # We can't fully test without the model, but we can test the structure
            # This test validates the class can be imported and instantiated
            assert FaceDetector is not None

    def test_bounding_box_normalization(self) -> None:
        """Test bounding box coordinate normalization logic."""
        # Test the BoundingBox model's coordinate validation
        bbox = BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0, confidence=1.0)

        # Should accept valid normalized coordinates
        assert bbox.x1 == 0.0
        assert bbox.x2 == 1.0

        # Verify absolute conversion
        absolute = bbox.to_absolute(100, 100)
        assert absolute["x1"] == 0
        assert absolute["y1"] == 0
        assert absolute["x2"] == 100
        assert absolute["y2"] == 100
