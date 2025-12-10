from dataclasses import dataclass
import os
from typing import Optional

import cv2


from .errors import JSONValidationError
from .base_media import BaseMedia
from .frame_generator import FrameGenerator


@dataclass
class ImageGenerator(BaseMedia):
    frame: Optional[FrameGenerator] = None

    @classmethod
    def from_dict(cls, out_dir:str,  data: dict):

        processedData = data.copy()
        processedData["out_dir"] = out_dir
        # Process base fields first, get them as a dictionary
        base_fields_dict = BaseMedia.from_dict(processedData).__dict__

        # Process frame field
        if "frame" in data and isinstance(data["frame"], dict):
            frame_instance = FrameGenerator.from_dict(data["frame"])
        elif "frame" not in data:
            raise JSONValidationError("ImageDescription missing 'frame' data.")
        else:
            raise JSONValidationError("Invalid 'frame' data for ImageDescription.")

        # Construct a dictionary with all arguments for ImageDescription's __init__
        # This ensures correct argument passing order for the dataclass constructor.
        init_args = {
            **base_fields_dict,  # All fields from BaseMedia
            "frame": frame_instance,  # Specific field for ImageDescription
        }

        # Filter to ensure only valid fields for ImageDescription are passed
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_init_args = {k: v for k, v in init_args.items() if k in valid_keys}

        return cls(**filtered_init_args)

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["frame"] = self.frame.to_dict()
        return data

    def generate(self):
        frame = self.frame.generate(self.width, self.height)
        cv2.imwrite(self.temp_filepath, frame)
        print(f"Image '{self.fileName}' created by OpenCV.")
        self.update_metadata()
        os.rename(self.temp_filepath, self.filepath)
        

