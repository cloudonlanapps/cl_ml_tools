"""Face detection task implementation."""

import json
import logging
from typing import Callable, override

from PIL import Image

from ...common.compute_module import ComputeModule
from ...common.job_storage import JobStorage
from .algo.face_detector import FaceDetector
from .schema import BoundingBox, FaceDetectionOutput, FaceDetectionParams

logger = logging.getLogger(__name__)


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

        with Image.open(input_path) as img:
            image_width, image_height = img.size

        faces = [
            BoundingBox(
                x1=det["x1"],
                y1=det["y1"],
                x2=det["x2"],
                y2=det["y2"],
                confidence=det["confidence"],
            )
            for det in detections
        ]

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
