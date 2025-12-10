from dataclasses import dataclass, field
from typing import List

from .base_media import BaseMedia, SupportedMIME
from .image_generator import ImageGenerator
from .video_generator import VideoGenerator
from .errors import JSONValidationError


@dataclass
class RandomMediaGenerator:
    """A collection of various media descriptions."""

    out_dir: str = None
    media_list: List[BaseMedia] = field(default_factory=list)

    @classmethod
    def supportedMIME(cls) -> list[str]:
        return SupportedMIME.MIME_TYPES.keys()

    @classmethod
    def from_dict(cls, outdir: str, data: dict):
        if (
            not isinstance(data, dict)
            or "media_list" not in data
            or not isinstance(data["media_list"], list)
        ):
            raise JSONValidationError(
                "Invalid input data for MediaGenerator. Expected a dict with 'media_list' key."
            )

        parsed_media_list: List[BaseMedia] = []
        for item_data in data["media_list"]:
            if not isinstance(item_data, dict):
                raise JSONValidationError(
                    "Invalid item in media_list. Expected a dictionary."
                )

            mime_type = item_data.get("MIMEType", "")
            if mime_type.startswith("image/"):
                parsed_media_list.append(ImageGenerator.from_dict(outdir, item_data))
            elif mime_type.startswith("video/"):
                parsed_media_list.append(VideoGenerator.from_dict(outdir, item_data))
            else:
                # Fallback to BaseMediaDescription if type is unknown or not image/video
                parsed_media_list.append(BaseMedia.from_dict(item_data))

        return cls(media_list=parsed_media_list)

    def to_dict(self) -> dict:
        return {"media_list": [media.to_dict() for media in self.media_list]}

    def generate(self):
        for media in self.media_list:
            media.generate()
