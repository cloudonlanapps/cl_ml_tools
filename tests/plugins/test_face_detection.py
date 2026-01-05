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
    from cl_ml_tools.common.job_storage import JobStorage, SavedJobFile
    from cl_ml_tools.common.job_repository import JobRepository

from cl_ml_tools.plugins.face_detection.algo.face_detector import FaceDetector
from cl_ml_tools.plugins.face_detection.schema import (
    BBox,
    DetectedFace,
    FaceDetectionOutput,
    FaceDetectionParams,
    FaceLandmarks,
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


def test_bbox_schema_validation():
    """Test BBox schema validates correctly."""
    bbox = BBox(x1=0.1, y1=0.2, x2=0.8, y2=0.9)

    assert bbox.x1 == 0.1
    assert bbox.y1 == 0.2
    assert bbox.x2 == 0.8
    assert bbox.y2 == 0.9


def test_detected_face_to_absolute():
    """Test DetectedFace.to_absolute converts coordinates."""
    bbox = BBox(x1=0.1, y1=0.2, x2=0.8, y2=0.9)
    landmarks = FaceLandmarks(
        right_eye=(0.2, 0.3),
        left_eye=(0.7, 0.3),
        nose_tip=(0.5, 0.5),
        mouth_right=(0.3, 0.7),
        mouth_left=(0.6, 0.7),
    )
    face = DetectedFace(
        bbox=bbox, 
        confidence=0.9, 
        landmarks=landmarks, 
        file_path="faces/face_0.png"
    )

    absolute = face.to_absolute(image_width=1000, image_height=800)

    # BBox check
    assert absolute["x1"] == 100
    assert absolute["y1"] == 160
    assert absolute["x2"] == 800
    assert absolute["y2"] == 720
    
    # Landmarks check
    assert absolute["landmarks"]["right_eye"] == (200, 240)
    assert absolute["landmarks"]["nose_tip"] == (500, 400)
    
    # File path check
    assert absolute["file_path"] == "faces/face_0.png"


def test_face_detection_output_schema_validation():
    """Test FaceDetectionOutput schema validates correctly."""
    bbox = BBox(x1=0.1, y1=0.1, x2=0.5, y2=0.5)
    landmarks = FaceLandmarks(
        right_eye=(0.2, 0.2),
        left_eye=(0.4, 0.2),
        nose_tip=(0.3, 0.3),
        mouth_right=(0.25, 0.4),
        mouth_left=(0.35, 0.4),
    )
    face = DetectedFace(
        bbox=bbox, 
        confidence=0.9, 
        landmarks=landmarks, 
        file_path="faces/face_0.png"
    )

    output = FaceDetectionOutput(
        faces=[face],
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


# ============================================================================
# ALGORITHM TESTS
# ============================================================================


# Note: Detailed unit tests for preprocess/postprocess were removed as they were
# implementation details of the previous algorithm. The new implementation
# relies on cv2.FaceDetectorYN which is tested via integration tests below.


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
        assert hasattr(face.bbox, "x1")
        assert getattr(face, "confidence", None) is not None
        assert getattr(face, "landmarks", None) is not None


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
            self,
            job_id: str,
            relative_path: str,
            file: Any,
            *,
            mkdirs: bool = True,
        ) -> "SavedJobFile":
            from cl_ml_tools.common.job_storage import SavedJobFile

            return SavedJobFile(relative_path=relative_path, size=0, hash=None)

        async def open(self, job_id: str, relative_path: str) -> Any:
            return None

        def resolve_path(self, job_id: str, relative_path: str | None = None) -> Path:
            if relative_path:
                return tmp_path / job_id / relative_path
            return tmp_path / job_id

        def allocate_path(
            self, job_id: str, relative_path: str, *, mkdirs: bool = True
        ) -> Path:
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

    # Verify output faces
    for i, face in enumerate(output.faces):
        assert face.file_path == f"faces/face_{i}.png"
        face_file = tmp_path / job_id / "faces" / f"face_{i}.png"
        assert face_file.exists(), f"Face file {face_file} does not exist"
        # Check if valid image
        with Image.open(face_file) as f_img:
            assert f_img.width > 0
            assert f_img.height > 0

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
            self,
            job_id: str,
            relative_path: str,
            file: Any,
            *,
            mkdirs: bool = True,
        ) -> Any:
            return None

        async def open(self, job_id: str, relative_path: str) -> Any:
            return None

        def resolve_path(self, job_id: str, relative_path: str | None = None) -> Path:
            return tmp_path / job_id / (relative_path or "")

        def allocate_path(
            self, job_id: str, relative_path: str, *, mkdirs: bool = True
        ) -> Path:
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
def test_face_detection_route_job_submission(
    api_client: "TestClient", sample_image_path: Path
):
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
