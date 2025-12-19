"""Unit and integration tests for face detection plugin.

Tests schema validation, bounding box detection, confidence thresholds, task execution, routes, and full job lifecycle.
Requires ML models downloaded.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from cl_ml_tools import Worker
    from cl_ml_tools.common.file_storage import JobStorage, SavedJobFile
    from cl_ml_tools.common.job_repository import JobRepository

from cl_ml_tools.plugins.face_detection.algo.face_detector import FaceDetector
from cl_ml_tools.plugins.face_detection.schema import (
    BoundingBox,
    FaceDetectionOutput,
    FaceDetectionParams,
)
from cl_ml_tools.plugins.face_detection.task import FaceDetectionTask

# ============================================================================
# SCHEMA TESTS
# ============================================================================


def test_face_detection_params_schema_validation():
    """Test FaceDetectionParams schema validates correctly."""
    params = FaceDetectionParams(
        input_path="/path/to/input.jpg",
        output_path="output/faces.json",
        confidence_threshold=0.8,
        nms_threshold=0.5,
    )

    assert params.input_path == "/path/to/input.jpg"
    assert params.output_path == "output/faces.json"
    assert params.confidence_threshold == 0.8
    assert params.nms_threshold == 0.5


def test_face_detection_params_defaults():
    """Test FaceDetectionParams has correct default values."""
    params = FaceDetectionParams(
        input_path="/path/to/input.jpg",
        output_path="output/faces.json",
    )

    assert params.confidence_threshold == 0.7
    assert params.nms_threshold == 0.4


def test_face_detection_params_threshold_validation():
    """Test FaceDetectionParams validates threshold ranges."""
    # Valid thresholds
    params = FaceDetectionParams(
        input_path="/path/to/input.jpg",
        output_path="output/faces.json",
        confidence_threshold=0.5,
        nms_threshold=0.3,
    )
    assert params.confidence_threshold == 0.5

    # Invalid confidence (too high)
    with pytest.raises(ValueError):
        _ = FaceDetectionParams(
            input_path="/path/to/input.jpg",
            output_path="output/faces.json",
            confidence_threshold=1.5,
        )

    # Invalid nms (negative)
    with pytest.raises(ValueError):
        _ = FaceDetectionParams(
            input_path="/path/to/input.jpg",
            output_path="output/faces.json",
            nms_threshold=-0.1,
        )


def test_bounding_box_schema_validation():
    """Test BoundingBox schema validates correctly."""
    bbox = BoundingBox(
        x1=0.1,
        y1=0.2,
        x2=0.8,
        y2=0.9,
        confidence=0.95,
    )

    assert bbox.x1 == 0.1
    assert bbox.y1 == 0.2
    assert bbox.x2 == 0.8
    assert bbox.y2 == 0.9
    assert bbox.confidence == 0.95


def test_bounding_box_to_absolute():
    """Test BoundingBox.to_absolute converts coordinates."""
    bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.8, y2=0.9, confidence=0.9)

    absolute = bbox.to_absolute(image_width=1000, image_height=800)

    assert absolute["x1"] == 100
    assert absolute["y1"] == 160
    assert absolute["x2"] == 800
    assert absolute["y2"] == 720


def test_face_detection_output_schema_validation():
    """Test FaceDetectionOutput schema validates correctly."""
    bbox = BoundingBox(x1=0.1, y1=0.1, x2=0.5, y2=0.5, confidence=0.9)

    output = FaceDetectionOutput(
        faces=[bbox],
        num_faces=1,
        image_width=800,
        image_height=600,
    )

    assert len(output.faces) == 1
    assert output.num_faces == 1
    assert output.image_width == 800
    assert output.image_height == 600


# ============================================================================
# ALGORITHM TESTS
# ============================================================================


@pytest.fixture
def mock_face_detector():
    """Fixture for FaceDetector with mocked ONNX session."""
    with patch(
        "cl_ml_tools.plugins.face_detection.algo.face_detector.ort.InferenceSession"
    ) as mock_sess:
        # Mock session input/output names
        mock_instance = mock_sess.return_value
        mock_instance.get_inputs.return_value = [MagicMock(name="input")]
        mock_instance.get_outputs.return_value = [MagicMock(name="boxes"), MagicMock(name="scores")]

        with patch("cl_ml_tools.plugins.face_detection.algo.face_detector.get_model_downloader"):
            with patch(
                "cl_ml_tools.plugins.face_detection.algo.face_detector.Path.exists",
                return_value=True,
            ):
                return FaceDetector(model_path="dummy.onnx")


def test_face_detector_preprocess(mock_face_detector):
    """Test image preprocessing."""
    img = Image.new("RGB", (1000, 800), color="red")
    input_array, original_size = mock_face_detector.preprocess(img)  # pyright: ignore[reportUnknownVariableType]

    assert original_size == (1000, 800)
    assert input_array.shape == (1, 3, 224, 224)  # pyright: ignore[reportUnknownMemberType]
    assert input_array.dtype == np.float32
    assert np.max(input_array) <= 1.0


def test_face_detector_calculate_iou(mock_face_detector):
    """Test IoU calculation."""
    # Box: [x_center, y_center, width, height]
    box1 = np.array([0.5, 0.5, 0.2, 0.2], dtype=np.float32)
    boxes = np.array(
        [
            [0.5, 0.5, 0.2, 0.2],  # Identical: IoU = 1.0
            [0.6, 0.5, 0.2, 0.2],  # Half overlap horizontally: IoU = 0.33...
            [0.8, 0.8, 0.2, 0.2],  # No overlap: IoU = 0.0
        ],
        dtype=np.float32,
    )

    ious = mock_face_detector._calculate_iou(box1, boxes)  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]

    assert ious[0] == pytest.approx(1.0, abs=1e-4)
    assert ious[1] == pytest.approx(0.333333, abs=1e-4)
    assert ious[2] == pytest.approx(0.0, abs=1e-4)


def test_face_detector_nms(mock_face_detector):
    """Test Non-Maximum Suppression."""
    boxes = np.array(
        [
            [0.5, 0.5, 0.2, 0.2],
            [0.51, 0.51, 0.2, 0.2],  # Overlaps boxes[0]
            [0.8, 0.8, 0.1, 0.1],  # Distinct
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.85, 0.8], dtype=np.float32)

    keep_indices = mock_face_detector._nms(boxes, scores, iou_threshold=0.5)  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]

    assert len(keep_indices) == 2
    assert 0 in keep_indices  # Kept boxes[0] because it has higher score
    assert 2 in keep_indices  # Kept boxes[2] because it doesn't overlap
    assert 1 not in keep_indices  # Suppressed boxes[1]


def test_face_detector_postprocess_center_format(mock_face_detector):
    """Test postprocessing with center format [x_center, y_center, width, height]."""
    # Normalized coords [0, 1]
    boxes = np.array([[[0.5, 0.5, 0.2, 0.2]]], dtype=np.float32)
    scores = np.array([[[0.9]]], dtype=np.float32)

    original_size = (1000, 1000)
    detections: list[dict[str, Any]] = mock_face_detector.postprocess(
        [boxes, scores], original_size
    )

    assert len(detections) == 1
    det = detections[0]
    # x_center=0.5, width=0.2 -> x1=0.4, x2=0.6
    assert det["x1"] == pytest.approx(400)
    assert det["y1"] == pytest.approx(400)
    assert det["x2"] == pytest.approx(600)
    assert det["y2"] == pytest.approx(600)
    assert det["confidence"] == pytest.approx(0.9)


def test_face_detector_postprocess_corner_format(mock_face_detector):
    """Test postprocessing with corner format [x1, y1, x2, y2]."""
    # If box[2] > 1.0 or box[3] > 1.0, it's considered corner format
    # Wait, the code says: if box[2] <= 1.0 and box[3] <= 1.0: normalized center
    # else: [x1, y1, x2, y2]

    boxes = np.array([[[0.1, 0.2, 0.3, 0.4]]], dtype=np.float32)  # Wait, 0.3 <= 1.0
    # Let's use > 1.0 to trigger corner format
    boxes = np.array([[[0.1, 0.2, 1.1, 1.2]]], dtype=np.float32)
    scores = np.array([[[0.8]]], dtype=np.float32)

    original_size = (1000, 1000)
    detections: list[dict[str, Any]] = mock_face_detector.postprocess(
        [boxes, scores], original_size
    )

    assert len(detections) == 1
    det = detections[0]
    # x1=0.1*1000=100, y1=0.2*1000=200, x2=1.1*1000=1100, y2=1.2*1000=1200
    assert det["x1"] == pytest.approx(100)
    assert det["y1"] == pytest.approx(200)
    assert det["x2"] == pytest.approx(1100)
    assert det["y2"] == pytest.approx(1200)


def test_face_detector_postprocess_empty_outputs(mock_face_detector):
    """Test postprocessing with empty outputs."""
    assert mock_face_detector.postprocess([], (100, 100)) == []
    assert mock_face_detector.postprocess([np.array([])], (100, 100)) == []


def test_face_detector_postprocess_low_confidence(mock_face_detector):
    """Test postprocessing filters low confidence detections."""
    boxes = np.array([[[0.5, 0.5, 0.2, 0.2]]], dtype=np.float32)
    scores = np.array([[[0.1]]], dtype=np.float32)  # Below 0.7

    detections: list[dict[str, Any]] = mock_face_detector.postprocess([boxes, scores], (100, 100))
    assert len(detections) == 0


def test_face_detector_preprocess_non_rgb(mock_face_detector):
    """Test preprocessing with non-RGB image."""
    img = Image.new("L", (100, 100), color=128)  # Greyscale
    input_array, original_size = mock_face_detector.preprocess(img)

    assert original_size == (100, 100)
    assert input_array.shape == (1, 3, 224, 224)


def test_face_detector_postprocess_single_output(mock_face_detector):
    """Test postprocessing with insufficient outputs."""
    outputs = [np.array([[[0.5, 0.5, 0.2, 0.2]]], dtype=np.float32)]
    detections: list[dict[str, Any]] = mock_face_detector.postprocess(outputs, (100, 100))
    assert detections == []


def test_face_detector_postprocess_no_boxes_after_threshold(mock_face_detector):
    """Test postprocessing when no boxes remain after confidence thresholding."""
    boxes = np.array([[[0.5, 0.5, 0.2, 0.2]]], dtype=np.float32)
    scores = np.array([[[0.1]]], dtype=np.float32)

    detections = mock_face_detector.postprocess([boxes, scores], (100, 100))
    assert detections == []


@pytest.mark.requires_models
def test_face_detection_algo_basic(sample_image_path: Path):
    """Test basic face detection."""
    from cl_ml_tools.plugins.face_detection.algo.face_detector import FaceDetector

    detector = FaceDetector()
    faces = detector.detect(str(sample_image_path), confidence_threshold=0.5)

    assert isinstance(faces, list)
    # May or may not contain faces depending on test image


@pytest.mark.requires_models
def test_face_detection_algo_returns_bounding_boxes(sample_image_path: Path):
    """Test face detection returns proper bounding box format."""
    from cl_ml_tools.plugins.face_detection.algo.face_detector import FaceDetector

    detector = FaceDetector()
    faces = detector.detect(str(sample_image_path), confidence_threshold=0.3)

    for face in faces:
        # Each face should have bbox coordinates and confidence
        assert "x1" in face or "bbox" in face or hasattr(face, "x1")


@pytest.mark.requires_models
def test_face_detection_algo_confidence_threshold(sample_image_path: Path):
    """Test confidence threshold filtering."""
    from cl_ml_tools.plugins.face_detection.algo.face_detector import FaceDetector

    detector = FaceDetector()

    # Low threshold - more detections
    faces_low = detector.detect(str(sample_image_path), confidence_threshold=0.3)

    # High threshold - fewer detections
    faces_high = detector.detect(str(sample_image_path), confidence_threshold=0.9)

    # High threshold should have same or fewer detections
    assert len(faces_high) <= len(faces_low)


@pytest.mark.requires_models
def test_face_detection_algo_error_handling_invalid_file(tmp_path: Path):
    """Test face detection handles invalid image files."""
    from cl_ml_tools.plugins.face_detection.algo.face_detector import FaceDetector

    invalid_file = tmp_path / "invalid.jpg"
    _ = invalid_file.write_text("not an image")

    detector = FaceDetector()

    with pytest.raises(Exception):
        detector.detect(str(invalid_file), confidence_threshold=0.5)


# ============================================================================
# TASK TESTS
# ============================================================================


@pytest.mark.requires_models
@pytest.mark.asyncio
async def test_face_detection_task_run_success(sample_image_path: Path, tmp_path: Path):
    """Test FaceDetectionTask execution success."""
    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(sample_image_path.read_bytes())

    params = FaceDetectionParams(
        input_path=str(input_path),
        output_path="output/faces.json",
        confidence_threshold=0.5,
    )

    task = FaceDetectionTask()
    task.setup()
    job_id = "test-job-123"

    class MockStorage:
        def create_directory(self, job_id: str) -> None:
            (tmp_path / job_id).mkdir(parents=True, exist_ok=True)

        def remove(self, job_id: str) -> bool:
            return True

        async def save(
            self, job_id: str, relative_path: str, file: Any, *, mkdirs: bool = True
        ) -> "SavedJobFile":
            from cl_ml_tools.common.file_storage import SavedJobFile

            return SavedJobFile(relative_path=relative_path, size=0, hash=None)

        async def open(self, job_id: str, relative_path: str) -> Any:
            return None

        def resolve_path(self, job_id: str, relative_path: str | None = None) -> Path:
            if relative_path:
                return tmp_path / job_id / relative_path
            return tmp_path / job_id

        def allocate_path(self, job_id: str, relative_path: str, *, mkdirs: bool = True) -> Path:
            output_path = tmp_path / job_id / relative_path
            if mkdirs:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            return output_path

    storage = MockStorage()

    output = await task.run(job_id, params, storage)

    assert isinstance(output, FaceDetectionOutput)
    assert output.num_faces >= 0
    assert output.image_width > 0
    assert output.image_height > 0

    # Verify output file
    output_file = tmp_path / job_id / "output" / "faces.json"
    assert output_file.exists()

    with open(output_file) as f:
        result = json.load(f)

    assert "num_faces" in result
    assert "faces" in result


@pytest.mark.requires_models
@pytest.mark.asyncio
async def test_face_detection_task_run_file_not_found(tmp_path: Path):
    """Test FaceDetectionTask raises FileNotFoundError for missing input."""
    params = FaceDetectionParams(
        input_path="/nonexistent/file.jpg",
        output_path="output/faces.json",
    )

    task = FaceDetectionTask()
    task.setup()
    job_id = "test-job-789"

    class MockStorage:
        def create_directory(self, job_id: str) -> None:
            pass

        def remove(self, job_id: str) -> bool:
            return True

        async def save(
            self, job_id: str, relative_path: str, file: Any, *, mkdirs: bool = True
        ) -> Any:
            return None

        async def open(self, job_id: str, relative_path: str) -> Any:
            return None

        def resolve_path(self, job_id: str, relative_path: str | None = None) -> Path:
            return tmp_path / job_id / (relative_path or "")

        def allocate_path(self, job_id: str, relative_path: str, *, mkdirs: bool = True) -> Path:
            return tmp_path / "output" / relative_path

    storage = MockStorage()

    with pytest.raises(FileNotFoundError):
        await task.run(job_id, params, storage)


# ============================================================================
# ROUTE TESTS
# ============================================================================


def test_face_detection_route_creation(api_client: "TestClient"):
    """Test face_detection route is registered."""
    response = api_client.get("/openapi.json")
    assert response.status_code == 200

    openapi = response.json()
    assert "/jobs/face_detection" in openapi["paths"]


@pytest.mark.requires_models
def test_face_detection_route_job_submission(api_client: "TestClient", sample_image_path: Path):
    """Test job submission via face_detection route."""
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/face_detection",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"confidence_threshold": "0.7", "priority": "5"},
        )

    assert response.status_code == 200
    data = response.json()

    assert "job_id" in data
    assert data["status"] == "queued"
    assert data["task_type"] == "face_detection"


# ============================================================================
# INTEGRATION TEST (API → Worker)
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_models
async def test_face_detection_full_job_lifecycle(
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
            "/jobs/face_detection",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"confidence_threshold": "0.5", "priority": "5"},
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
    output = FaceDetectionOutput.model_validate(job.output)
    assert output.num_faces >= 0
    assert len(output.faces) == output.num_faces
