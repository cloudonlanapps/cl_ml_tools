"""Face detection task implementation."""

import json
from typing import Callable, cast, override

import cv2
import numpy as np
from loguru import logger

from ...common.compute_module import ComputeModule
from ...common.job_storage import JobStorage
from ...utils.profiling import timed
from .algo.face_aligner import align_and_crop
from .algo.face_detector import FaceDetector
from .schema import (
    BBox,
    DetectedFace,
    FaceDetectionOutput,
    FaceDetectionParams,
    FaceLandmarks,
)


class FaceDetectionTask(ComputeModule[FaceDetectionParams, FaceDetectionOutput]):
    """Compute module for detecting faces in an image using ONNX model."""

    schema: type[FaceDetectionParams] = FaceDetectionParams

    def __init__(self) -> None:
        self._detector: FaceDetector | None = None

    @property
    @override
    def task_type(self) -> str:
        return "face_detection"

    @override
    def setup(self) -> None:
        if self._detector is None:
            try:
                self._detector = FaceDetector()
                logger.info("Face detector initialized successfully")
            except (FileNotFoundError, RuntimeError, ImportError, OSError) as exc:
                logger.error("Face detector initialization failed", exc_info=exc)
                raise RuntimeError(
                    "Failed to initialize face detector. "
                    + "Ensure ONNX Runtime is installed and the model is available."
                ) from exc

    @override
    @timed
    async def run(
        self,
        job_id: str,
        params: FaceDetectionParams,
        storage: JobStorage,
        progress_callback: Callable[[int], None] | None = None,
    ) -> FaceDetectionOutput:
        if not self._detector:
            raise RuntimeError("Face detector is not initialized")

        input_path = storage.resolve_path(job_id, params.input_path)

        detections = self._detector.detect(
            image_path=str(input_path),
            confidence_threshold=params.confidence_threshold,
            nms_threshold=params.nms_threshold,
        )

        # Read image for alignment
        img_cv = cv2.imread(str(input_path))
        if img_cv is None:
            raise RuntimeError(f"Failed to read image with OpenCV: {input_path}")

        # Get dimensions for normalization (could also use img_cv.shape)
        image_height, image_width = cast(list[int], img_cv.shape[:2])

        faces: list[DetectedFace] = []
        for i, det in enumerate(detections):
            # Normalize bounding box
            bbox = BBox(
                x1=max(0.0, min(1.0, det.bbox.x1 / image_width)),
                y1=max(0.0, min(1.0, det.bbox.y1 / image_height)),
                x2=max(0.0, min(1.0, det.bbox.x2 / image_width)),
                y2=max(0.0, min(1.0, det.bbox.y2 / image_height)),
            )

            # Normalize landmarks
            landmarks = FaceLandmarks(
                right_eye=(
                    det.landmarks.right_eye[0] / image_width,
                    det.landmarks.right_eye[1] / image_height,
                ),
                left_eye=(
                    det.landmarks.left_eye[0] / image_width,
                    det.landmarks.left_eye[1] / image_height,
                ),
                nose_tip=(
                    det.landmarks.nose_tip[0] / image_width,
                    det.landmarks.nose_tip[1] / image_height,
                ),
                mouth_right=(
                    det.landmarks.mouth_right[0] / image_width,
                    det.landmarks.mouth_right[1] / image_height,
                ),
                mouth_left=(
                    det.landmarks.mouth_left[0] / image_width,
                    det.landmarks.mouth_left[1] / image_height,
                ),
            )

            # Align and crop face
            # Landmark order: [Right Eye, Left Eye, Nose, Right Mouth, Left Mouth]
            lm_points = [
                det.landmarks.right_eye,
                det.landmarks.left_eye,
                det.landmarks.nose_tip,
                det.landmarks.mouth_right,
                det.landmarks.mouth_left,
            ]

            try:
                aligned_face, _ = align_and_crop(img_cv.astype(np.uint8), lm_points)

                # Save cropped face
                face_relative_path = f"faces/face_{i}.png"
                face_abs_path = storage.allocate_path(job_id, face_relative_path)
                _ = cv2.imwrite(str(face_abs_path), aligned_face)

                faces.append(
                    DetectedFace(
                        bbox=bbox,
                        confidence=det.confidence,
                        landmarks=landmarks,
                        file_path=face_relative_path,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to align/crop face {i}: {e}")
                # Should we skip or include without file_path?
                # Schema requires file_path. For now, skipping failed alignments or failing task?
                # Given strict schema, failing or skipping is needed.
                # Let's skip safely but log error, or maybe raise if critical.
                # Requirement implies we MUST save.
                raise RuntimeError(f"Face alignment failed for face {i}") from e

        output = FaceDetectionOutput(
            faces=faces,
            num_faces=len(faces),
            image_width=image_width,
            image_height=image_height,
        )

        path = storage.allocate_path(
            job_id=job_id,
            relative_path=params.output_path,
        )

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output.model_dump(), f, indent=2)

        if progress_callback:
            progress_callback(100)

        return output
