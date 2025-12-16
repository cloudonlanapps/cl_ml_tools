"""EXIF metadata extraction plugin."""

from .schema import ExifMetadata, ExifParams
from .task import ExifTask

__all__ = ["ExifTask", "ExifParams", "ExifMetadata"]
