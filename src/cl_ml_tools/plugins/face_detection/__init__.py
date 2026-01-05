"""Face detection plugin using ONNX models."""

from .schema import BBox, DetectedFace, FaceDetectionOutput, FaceDetectionParams, FaceLandmarks
from .task import FaceDetectionTask

__all__ = [
    "FaceDetectionTask",
    "FaceDetectionParams",
    "FaceDetectionOutput",
    "DetectedFace",
    "BBox",
    "FaceLandmarks",
]
