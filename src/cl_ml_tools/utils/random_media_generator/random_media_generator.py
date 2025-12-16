from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from .base_media import BaseMedia, SupportedMIME
from .errors import JSONValidationError
from .image_generator import ImageGenerator
from .video_generator import VideoGenerator

RawMedia = Mapping[str, object]
RawMediaList = Sequence[RawMedia]


class RandomMediaGenerator(BaseModel):
    """A collection of various media descriptions."""

    out_dir: str
    media_list: list[BaseMedia] = Field(default_factory=list)

    # ----------------------------
    # Class helpers
    # ----------------------------

    @classmethod
    def supportedMIME(cls) -> list[str]:
        return list(SupportedMIME.MIME_TYPES.keys())

    # ----------------------------
    # Validators
    # ----------------------------

    @field_validator("media_list", mode="before")
    @classmethod
    def validate_media_list(cls, v: RawMediaList, info: ValidationInfo):
        out_dir = info.data.get("out_dir")
        if not isinstance(out_dir, str):
            raise JSONValidationError("'out_dir' must be provided before 'media_list'.")

        parsed: list[BaseMedia] = []

        for item in v:
            mime_type_obj = item.get("MIMEType")
            mime_type = mime_type_obj if isinstance(mime_type_obj, str) else ""

            if mime_type.startswith("image/"):
                parsed.append(ImageGenerator.model_validate({**item, "out_dir": out_dir}))
            elif mime_type.startswith("video/"):
                parsed.append(VideoGenerator.model_validate({**item, "out_dir": out_dir}))
            else:
                parsed.append(BaseMedia.model_validate({**item, "out_dir": out_dir}))

        return parsed
