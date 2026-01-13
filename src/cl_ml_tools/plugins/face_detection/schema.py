"""Face detection parameters and output schemas."""

from typing import override

from pydantic import BaseModel, Field

from ...common.schema_job import BaseJobParams, TaskOutput


class FaceDetectionParams(BaseJobParams):
    """Parameters for face detection task."""

    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for face detections (0.0-1.0)",
    )
    nms_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Non-maximum suppression threshold for overlapping boxes (0.0-1.0)",
    )


class FaceLandmarks(BaseModel):
    """Facial landmarks with normalized coordinates [0.0, 1.0]."""

    right_eye: tuple[float, float] = Field(..., description="Right eye (x, y)")
    left_eye: tuple[float, float] = Field(..., description="Left eye (x, y)")
    nose_tip: tuple[float, float] = Field(..., description="Nose tip (x, y)")
    mouth_right: tuple[float, float] = Field(..., description="Right mouth corner (x, y)")
    mouth_left: tuple[float, float] = Field(..., description="Left mouth corner (x, y)")

    def as_ordered_list(self) -> list[list[float]]:
        """
        Return landmarks in the exact required order:
        [right_eye, left_eye, nose_tip, mouth_right, mouth_left]
        """
        return [
            [val for val in self.right_eye],
            [val for val in self.left_eye],
            [val for val in self.nose_tip],
            [val for val in self.mouth_right],
            [val for val in self.mouth_left],
        ]

    def to_absolute(self, width: int, height: int) -> "FaceLandmarks":
        return FaceLandmarks.model_validate(
            {
                "right_eye": list(
                    (int(self.right_eye[0] * width), int(self.right_eye[1] * height))
                ),
                "left_eye": list((int(self.left_eye[0] * width), int(self.left_eye[1] * height))),
                "nose_tip": list((int(self.nose_tip[0] * width), int(self.nose_tip[1] * height))),
                "mouth_right": list(
                    (int(self.mouth_right[0] * width), int(self.mouth_right[1] * height))
                ),
                "mouth_left": list(
                    (int(self.mouth_left[0] * width), int(self.mouth_left[1] * height))
                ),
            }
        )


class BBox(BaseModel):
    """Bounding box with normalized coordinates [0.0, 1.0]."""

    x1: float = Field(..., ge=0.0, le=1.0, description="Left coordinate (normalized)")
    y1: float = Field(..., ge=0.0, le=1.0, description="Top coordinate (normalized)")
    x2: float = Field(..., ge=0.0, le=1.0, description="Right coordinate (normalized)")
    y2: float = Field(..., ge=0.0, le=1.0, description="Bottom coordinate (normalized)")

    def as_ordered_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_absolute(self, width: int, height: int) -> "BBox":
        return BBox.model_validate(
            {
                "x1": int(self.x1 * width),
                "y1": int(self.y1 * height),
                "x2": int(self.x2 * width),
                "y2": int(self.y2 * height),
            }
        )


class DetectedFace(BaseModel):
    """Detected face with bounding box and landmarks."""

    bbox: BBox = Field(..., description="Face bounding box")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    landmarks: FaceLandmarks = Field(..., description="Detected facial landmarks")
    file_path: str = Field(..., description="Relative path to the cropped face image")

    @override
    def model_dump(self, **kwargs):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        """Override to ensure landmarks tuples are serialized as lists."""
        data = super().model_dump(**kwargs)  # pyright: ignore[reportUnknownArgumentType]
        # Ensure landmarks dict has lists instead of tuples
        data["bbox"] = self.bbox.as_ordered_list()
        data["landmarks"] = self.landmarks.as_ordered_list()
        return data

    def to_absolute(self, image_width: int, image_height: int) -> "DetectedFace":
        """Convert all coordinates to absolute pixel values."""
        return DetectedFace.model_validate(
            {
                "bbox": self.bbox.to_absolute(image_width, image_height).as_ordered_list(),
                "confidence": self.confidence,
                "landmarks": self.landmarks.to_absolute(
                    image_width, image_height
                ).as_ordered_list(),
                "file_path": self.file_path,
            }
        )


class FaceDetectionOutput(TaskOutput):
    faces: list[DetectedFace] = Field(default_factory=list, description="List of detected faces")
    num_faces: int = Field(..., description="Total number of faces detected")
    image_width: int = Field(..., description="Input image width in pixels")
    image_height: int = Field(..., description="Input image height in pixels")
