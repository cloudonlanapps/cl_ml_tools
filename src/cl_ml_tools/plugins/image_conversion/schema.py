"""Image conversion parameters schema."""

from typing import Literal

from pydantic import Field

from cl_ml_tools.common.schemas import BaseJobParams


class ImageConversionParams(BaseJobParams):
    """Parameters for image conversion task.

    Attributes:
        input_paths: List of absolute paths to input images
        output_paths: List of absolute paths for converted images
        format: Target image format (png, jpg, jpeg, webp, gif, bmp, tiff)
        quality: Output quality for lossy formats (1-100)
    """

    format: Literal["png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"] = Field(
        ..., description="Target image format"
    )
    quality: int = Field(
        default=85, ge=1, le=100, description="Output quality for lossy formats (1-100)"
    )
