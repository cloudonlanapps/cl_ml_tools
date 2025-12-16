"""Face detection task implementation."""

import logging
from typing import Callable, Literal, TypedDict, override

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from .algo.face_detector import FaceDetector
from .schema import BoundingBox, FaceDetectionParams, FaceDetectionResult

logger = logging.getLogger(__name__)


class FileSuccessResult(TypedDict):
    file_path: str
    status: Literal["success"]
    detection: dict[str, object]


class FileErrorResult(TypedDict):
    file_path: str
    status: Literal["error"]
    error: str


FileResult = FileSuccessResult | FileErrorResult


class FaceDetectionTask(ComputeModule[FaceDetectionParams]):
    """Compute module for detecting faces in images using ONNX model."""

    def __init__(self) -> None:
        super().__init__()
        self._detector: FaceDetector | None = None

    @property
    @override
    def task_type(self) -> str:
        return "face_detection"

    @override
    def get_schema(self) -> type[BaseJobParams]:
        return FaceDetectionParams

    def _get_detector(self) -> FaceDetector:
        if self._detector is None:
            self._detector = FaceDetector()
            logger.info("Face detector initialized successfully")
        return self._detector

    @override
    async def execute(
        self,
        job: Job,
        params: FaceDetectionParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        try:
            try:
                detector = self._get_detector()
            except Exception as e:
                logger.error("Face detector initialization failed: %s", e)
                return TaskResult(status = "error", error = (
                        "Failed to initialize face detector: "
                        f"{e}. Ensure ONNX Runtime is installed and the model can be downloaded."
                    ))

            file_results: list[FileResult] = []
            total_files: int = len(params.input_paths)

            from PIL import Image

            for index, input_path in enumerate(params.input_paths):
                try:
                    detections = detector.detect(
                        image_path=input_path,
                        confidence_threshold=params.confidence_threshold,
                        nms_threshold=params.nms_threshold,
                    )

                    with Image.open(input_path) as img:
                        image_width, image_height = img.size

                    face_boxes = [
                        BoundingBox(
                            x1=det["x1"],
                            y1=det["y1"],
                            x2=det["x2"],
                            y2=det["y2"],
                            confidence=det["confidence"],
                        )
                        for det in detections
                    ]

                    result = FaceDetectionResult(
                        file_path=input_path,
                        faces=face_boxes,
                        num_faces=len(face_boxes),
                        image_width=image_width,
                        image_height=image_height,
                    )

                    file_results.append(
                        {
                            "file_path": input_path,
                            "status": "success",
                            "detection": result.model_dump(),
                        }
                    )

                except FileNotFoundError:
                    logger.error("File not found: %s", input_path)
                    file_results.append(
                        {
                            "file_path": input_path,
                            "status": "error",
                            "error": "File not found",
                        }
                    )

                except Exception as e:
                    logger.error("Failed to detect faces in %s: %s", input_path, e)
                    file_results.append(
                        {
                            "file_path": input_path,
                            "status": "error",
                            "error": str(e),
                        }
                    )

                if progress_callback:
                    progress_callback(int((index + 1) / total_files * 100))

            all_success: bool = all(r["status"] == "success" for r in file_results)
            any_success: bool = any(r["status"] == "success" for r in file_results)

            if not any_success:
                return TaskResult(status = "error", task_output = {
                        "files": file_results,
                        "total_files": total_files,
                    }, error = "Failed to detect faces in all files")

            if not all_success:
                success_count = sum(1 for r in file_results if r["status"] == "success")
                logger.warning(
                    "Partial success: %d/%d files processed successfully",
                    success_count,
                    total_files,
                )

            return TaskResult(status = "ok", task_output = {
                    "files": file_results,
                    "total_files": total_files,
                    "confidence_threshold": params.confidence_threshold,
                    "nms_threshold": params.nms_threshold,
                })

        except Exception as e:
            logger.exception("Unexpected error in FaceDetectionTask: %s", e)
            return TaskResult(status = "error", error = f"Task failed: {e}")
