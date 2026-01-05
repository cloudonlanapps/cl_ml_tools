"""Face detection using OpenCV YuNet model.

Model Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
Model File: face_detection_yunet_2023mar.onnx
"""

import logging
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from PIL import Image

from ....utils.model_downloader import get_model_downloader

logger = logging.getLogger(__name__)

# Model configuration
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
MODEL_SHA256: str | None = None


class FaceDetection(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


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
            
        height, width, _ = image.shape
        
        # Update detector input size
        self._detector.setInputSize((width, height))
        self._detector.setScoreThreshold(confidence_threshold)
        self._detector.setNMSThreshold(nms_threshold)
        
        # Run inference
        # faces: [1, num_faces, 15]
        # Format: x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, confidence
        _, faces = self._detector.detect(image)
        
        detections: list[FaceDetection] = []
        
        if faces is not None:
            for face in faces:
                x1, y1, w, h = face[0:4]
                confidence = face[14]
                
                detections.append(
                    FaceDetection(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x1 + w),
                        y2=float(y1 + h),
                        confidence=float(confidence),
                    )
                )

        logger.info(
            "Detected %d faces in %s",
            len(detections),
            image_path,
        )

        return detections
