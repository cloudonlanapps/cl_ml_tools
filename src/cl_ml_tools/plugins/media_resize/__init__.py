"""Media resize plugin."""

from .schema import MediaResizeParams
from .task import MediaResizeTask

__all__ = ["MediaResizeTask", "MediaResizeParams"]
