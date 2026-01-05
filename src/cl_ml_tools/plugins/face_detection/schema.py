"""Face detection parameters and output schemas."""

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

    model_config = {"ser_json_timedelta": "float", "ser_json_bytes": "base64"}

    right_eye: tuple[float, float] = Field(..., description="Right eye (x, y)")
    left_eye: tuple[float, float] = Field(..., description="Left eye (x, y)")
    nose_tip: tuple[float, float] = Field(..., description="Nose tip (x, y)")
    mouth_right: tuple[float, float] = Field(..., description="Right mouth corner (x, y)")
    mouth_left: tuple[float, float] = Field(..., description="Left mouth corner (x, y)")

    def model_dump(self, **kwargs):
        """Override to ensure tuples are serialized as lists for JSON compatibility."""
        data = super().model_dump(**kwargs)
        # Convert tuples to lists for JSON serialization
        for key in ['right_eye', 'left_eye', 'nose_tip', 'mouth_right', 'mouth_left']:
            if key in data and isinstance(data[key], tuple):
                data[key] = list(data[key])
        return data

    def to_absolute(self, width: int, height: int) -> dict[str, tuple[int, int]]:
        return {
            "right_eye": (int(self.right_eye[0] * width), int(self.right_eye[1] * height)),
            "left_eye": (int(self.left_eye[0] * width), int(self.left_eye[1] * height)),
            "nose_tip": (int(self.nose_tip[0] * width), int(self.nose_tip[1] * height)),
            "mouth_right": (int(self.mouth_right[0] * width), int(self.mouth_right[1] * height)),
            "mouth_left": (int(self.mouth_left[0] * width), int(self.mouth_left[1] * height)),
        }


class BBox(BaseModel):
    """Bounding box with normalized coordinates [0.0, 1.0]."""
    x1: float = Field(..., ge=0.0, le=1.0, description="Left coordinate (normalized)")
    y1: float = Field(..., ge=0.0, le=1.0, description="Top coordinate (normalized)")
    x2: float = Field(..., ge=0.0, le=1.0, description="Right coordinate (normalized)")
    y2: float = Field(..., ge=0.0, le=1.0, description="Bottom coordinate (normalized)")

    def to_absolute(self, width: int, height: int) -> dict[str, int]:
        return {
            "x1": int(self.x1 * width),
            "y1": int(self.y1 * height),
            "x2": int(self.x2 * width),
            "y2": int(self.y2 * height),
        }


class DetectedFace(BaseModel):
    """Detected face with bounding box and landmarks."""

    bbox: BBox = Field(..., description="Face bounding box")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    landmarks: FaceLandmarks = Field(..., description="Detected facial landmarks")
    file_path: str = Field(..., description="Relative path to the cropped face image")

    def model_dump(self, **kwargs):
        """Override to ensure landmarks tuples are serialized as lists."""
        data = super().model_dump(**kwargs)
        # Ensure landmarks dict has lists instead of tuples
        if 'landmarks' in data and isinstance(data['landmarks'], dict):
            landmarks_data = data['landmarks']
            for key in ['right_eye', 'left_eye', 'nose_tip', 'mouth_right', 'mouth_left']:
                if key in landmarks_data and isinstance(landmarks_data[key], tuple):
                    landmarks_data[key] = list(landmarks_data[key])
        return data

    def to_absolute(self, image_width: int, image_height: int) -> dict:
        """Convert all coordinates to absolute pixel values."""
        return {
            **self.bbox.to_absolute(image_width, image_height),
            "confidence": self.confidence,
            "landmarks": self.landmarks.to_absolute(image_width, image_height),
            "file_path": self.file_path,
        }


class FaceDetectionOutput(TaskOutput):
    faces: list[DetectedFace] = Field(
        default_factory=list, description="List of detected faces"
    )
    num_faces: int = Field(..., description="Total number of faces detected")
    image_width: int = Field(..., description="Input image width in pixels")
    image_height: int = Field(..., description="Input image height in pixels")
