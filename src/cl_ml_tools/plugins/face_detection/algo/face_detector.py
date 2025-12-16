"""Face detection using ONNX model (MediaPipe Face Detection).

Model Source: https://huggingface.co/qualcomm/MediaPipe-Face-Detection
Model File: MediaPipeFaceLandmarkDetector.onnx (2.45 MB)
"""

import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from PIL import Image

from ....utils.model_downloader import get_model_downloader

logger = logging.getLogger(__name__)

# Model configuration
MODEL_URL = (
    "https://huggingface.co/qualcomm/MediaPipe-Face-Detection/"
    "resolve/main/MediaPipeFaceLandmarkDetector.onnx"
)
MODEL_FILENAME = "mediapipe_face_detection.onnx"
MODEL_SHA256: str | None = None  # TODO: Add SHA256 hash for verification

# Expected input shape for MediaPipe Face Detection
INPUT_SIZE: tuple[int, int] = (192, 192)  # (height, width)


class FaceDetection(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class FaceDetector:
    """ONNX-based face detector using MediaPipe Face Detection model."""

    session: ort.InferenceSession
    input_name: str
    output_names: list[str]

    def __init__(self, model_path: str | Path | None = None) -> None:
        if model_path is None:
            downloader = get_model_downloader()
            logger.info("Downloading face detection model from %s", MODEL_URL)
            model_path = downloader.download(
                url=MODEL_URL,
                filename=MODEL_FILENAME,
                expected_sha256=MODEL_SHA256,
            )
        else:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info("Loading face detection model from %s", model_path)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        logger.info(
            "Model loaded. Input: %s, Outputs: %s",
            self.input_name,
            self.output_names,
        )

    def preprocess(self, image: Image.Image) -> tuple[NDArray[np.float32], tuple[int, int]]:
        original_size = image.size  # (width, height)

        if image.mode != "RGB":
            image = image.convert("RGB")

        image_resized = image.resize(
            (INPUT_SIZE[1], INPUT_SIZE[0]),
            Image.Resampling.BILINEAR,
        )

        img_array: NDArray[np.float32] = np.asarray(image_resized, dtype=np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

        return img_array, original_size

    def postprocess(
        self,
        outputs: list[NDArray[np.float32]],
        original_size: tuple[int, int],
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.4,
    ) -> list[FaceDetection]:
        _ = outputs
        _ = original_size
        _ = confidence_threshold
        _ = nms_threshold

        logger.warning("Face detection post-processing not implemented for this model.")

        return []

    def detect(
        self,
        image_path: str | Path,
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.4,
    ) -> list[FaceDetection]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path)

        input_array, original_size = self.preprocess(image)

        outputs: list[NDArray[np.float32]] = self.session.run(
            self.output_names,
            {self.input_name: input_array},
        )

        detections = self.postprocess(
            outputs,
            original_size,
            confidence_threshold,
            nms_threshold,
        )

        logger.info(
            "Detected %d faces in %s",
            len(detections),
            image_path,
        )

        return detections
