"""Media resize parameters schema."""

from ...common.schemas import BaseJobParams


class MediaResizeParams(BaseJobParams):
    """Parameters for image/video resize task.

    Supports both image and video resizing with flexible dimensions.
    Media type is auto-detected from file content.

    Attributes:
        input_paths: List of absolute paths to input media (images or videos)
        output_paths: List of absolute paths for resized output
        width: Target width in pixels (None = auto, default 256)
        height: Target height in pixels (None = auto, default 256)
        maintain_aspect_ratio: If True, maintain aspect ratio (default: True)
                               Note: For videos, aspect ratio is always maintained
        media_type: Detected media type (None initially, populated during execution)
    """

    width: int | None = None
    height: int | None = None
    maintain_aspect_ratio: bool = True
    media_type: str | None = None
