import os
from typing import override

import cv2
from pydantic import Field, field_validator, model_validator

from .base_media import BaseMedia
from .errors import JSONValidationError
from .frame_generator import FrameGenerator


class ImageGenerator(BaseMedia):
    frame: FrameGenerator | None = Field(default=None)

    @field_validator("frame", mode="before")
    @classmethod
    def validate_frame(cls, v: FrameGenerator | dict[str, str]):
        if isinstance(v, dict):
            return FrameGenerator.model_validate(v)

        return v

    @model_validator(mode="after")
    def ensure_frame_present(self):
        if self.frame is None:
            raise JSONValidationError("ImageDescription missing 'frame' data.")
        return self

    @override
    def generate(self) -> None:
        if self.frame is None:
            raise JSONValidationError("ImageDescription missing 'frame' data.")
        frame = self.frame.generate_frame(self.width, self.height)

        _ = cv2.imwrite(self.temp_filepath, frame)
        print(f"Image '{self.fileName}' created by OpenCV.")

        self.update_metadata()
        os.rename(self.temp_filepath, self.filepath)
