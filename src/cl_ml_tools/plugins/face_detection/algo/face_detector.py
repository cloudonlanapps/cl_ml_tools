"""Face detection using OpenCV YuNet model.

Model Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
Model File: face_detection_yunet_2023mar.onnx
"""

from pathlib import Path
from typing import cast

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel

from ....utils.model_downloader import get_model_downloader

# Model configuration
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
MODEL_SHA256: str | None = None


class FaceLandmarks(BaseModel):
    right_eye: tuple[float, float]
    left_eye: tuple[float, float]
    nose_tip: tuple[float, float]
    mouth_right: tuple[float, float]
    mouth_left: tuple[float, float]


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class FaceDetection(BaseModel):
    bbox: BBox
    confidence: float
    landmarks: FaceLandmarks


class FaceDetector:
    """Face detector using OpenCV's YuNet implementation."""

    _detector: cv2.FaceDetectorYN

    def __init__(self, model_path: str | Path | None = None) -> None:
        if model_path is None:
            downloader = get_model_downloader()
            logger.info("Downloading face detection model from %s", MODEL_URL)
            model_path = downloader.download(
                url=MODEL_URL,
                filename=MODEL_FILENAME,
                expected_sha256=MODEL_SHA256,
                auto_extract=False,
            )
        else:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info("Loading face detection model from %s", model_path)

        # Initialize YuNet
        # Input size will be set dynamically per image
        self._detector = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=(320, 320),  # Default, will be updated
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )

    def matlike_to_face_rows(self, mat: object) -> list[list[float]]:
        if mat is None:
            return []

        # YuNet returns a numpy array, usually (1, N, 15) or (N, 15)
        # We specify the type to avoid 'Unknown' shape warnings
        mat_array: np.ndarray[tuple[int, ...], np.dtype[np.float32]] = np.asarray(
            mat, dtype=np.float32
        )

        if mat_array.size == 0:
            return []

        # Squeeze batch dimension if present (1, N, 15) -> (N, 15)
        if mat_array.ndim == 3 and mat_array.shape[0] == 1:
            mat_array = mat_array.squeeze(0)
        elif mat_array.ndim == 1:
            # Single face (15,) -> (1, 15)
            mat_array = mat_array.reshape(1, -1)

        faces: list[list[float]] = []
        for row in mat_array:
            # Ensure we have the expected 15 values
            if len(row) != 15:
                # If it's still (1, N, 15) after squeeze it might look like this
                if row.ndim > 1:
                    for sub_row in row:
                        if len(sub_row) == 15:
                            faces.append([float(v) for v in sub_row])
                    continue
                raise ValueError(f"Expected 15 values per face, got {len(row)}")

            faces.append([float(v) for v in row])

        return faces

    def detect(
        self,
        image_path: str | Path,
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.4,
    ) -> list[FaceDetection]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load image with OpenCV to keep it BGR (what OpenCV expects usually)
        # But PIL logic passes image path. Let's use cv2.imread for robustness.
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        height, width, _ = cast(list[int], image.shape)

        # Update detector input size
        self._detector.setInputSize((width, height))
        self._detector.setScoreThreshold(confidence_threshold)
        self._detector.setNMSThreshold(nms_threshold)

        # Run inference
        # faces: [1, num_faces, 15]
        # Format: x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, confidence
        ret, faces_raw = self._detector.detect(image)
        if not ret:
            raise RuntimeError("Face detection failed")

        faces = self.matlike_to_face_rows(faces_raw)

        if len(faces) == 0:
            return []

        detections: list[FaceDetection] = []
        if not faces:
            return detections

        for face in faces:
            x1, y1, w, h = face[0:4]

            detections.append(
                FaceDetection(
                    bbox=BBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x1 + w),
                        y2=float(y1 + h),
                    ),
                    # Landmarks are pairs from index 4 to 13 (5 points)
                    # re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y
                    landmarks=FaceLandmarks(
                        right_eye=(float(face[4]), float(face[5])),
                        left_eye=(float(face[6]), float(face[7])),
                        nose_tip=(float(face[8]), float(face[9])),
                        mouth_right=(float(face[10]), float(face[11])),
                        mouth_left=(float(face[12]), float(face[13])),
                    ),
                    confidence=float(face[14]),
                )
            )

        logger.info(
            "Detected %d faces in %s",
            len(detections),
            image_path,
        )

        return detections
