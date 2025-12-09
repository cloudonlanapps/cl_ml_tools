"""Image resize parameters schema."""

from cl_ml_tools.common.schemas import BaseJobParams


class ImageResizeParams(BaseJobParams):
    """Parameters for image resize task.

    Attributes:
        input_paths: List of absolute paths to input images
        output_paths: List of absolute paths for resized images
        width: Target width in pixels
        height: Target height in pixels
        maintain_aspect_ratio: If True, maintain aspect ratio (default: False)
    """

    width: int
    height: int
    maintain_aspect_ratio: bool = False
