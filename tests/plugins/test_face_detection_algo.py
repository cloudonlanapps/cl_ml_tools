import os
from pathlib import Path

import pytest
from pydantic import BaseModel, TypeAdapter

from cl_ml_tools.plugins.face_detection.algo.face_detector import FaceDetector

CONFIDENCE_THRESHOLD = 0.5

# Standardize path resolution
TEST_VECTORS_DIR = Path(
    os.getenv("TEST_VECTORS_DIR", str(Path.home() / "cl_server_test_media"))
)


class FaceTestCase(BaseModel):
    path: str
    expected_count: int
    exact_match: bool


@pytest.mark.requires_models
class TestFaceDetectorIntegration:
    """Integration tests for FaceDetector using real images."""

    def test_face_detection(self):
        """Test detection on a list of images configured in JSON."""
        json_path = Path(__file__).parent / "test_face_detection.json"

        if not json_path.exists():
            pytest.fail(f"Test data file not found: {json_path}")

        # Parse JSON using Pydantic
        adapter = TypeAdapter(list[FaceTestCase])
        test_cases = adapter.validate_json(json_path.read_text())

        detector = FaceDetector()
        failures = []

        for case in test_cases:
            # Resolve relative to TEST_VECTORS_DIR
            image_path = TEST_VECTORS_DIR / case.path
            
            if not image_path.exists():
                failures.append(f"Image not found: {image_path} (from {case.path})")
                continue

            try:
                # Use global CONFIDENCE_THRESHOLD
                detections = detector.detect(
                    image_path, confidence_threshold=CONFIDENCE_THRESHOLD
                )
                count = len(detections)

                if case.exact_match:
                    if count != case.expected_count:
                        failures.append(
                            f"Case {image_path.name}: Expected exactly {case.expected_count} faces, found {count}"
                        )
                else:
                    if count < case.expected_count:
                        failures.append(
                            f"Case {image_path.name}: Expected at least {case.expected_count} faces, found {count}"
                        )

                # Check for landmarks in first detection if any found
                if count > 0:
                    first_det = detections[0]
                    # Check if landmarks attribute exists and has data (basic check)
                    if not first_det.landmarks:
                         failures.append(f"Case {image_path.name}: Landmarks missing in detection")
            except Exception as e:
                failures.append(
                    f"Case {image_path.name}: Exception during detection - {str(e)}"
                )

        if failures:
            pytest.fail("\n".join(failures))
