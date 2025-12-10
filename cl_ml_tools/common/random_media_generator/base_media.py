from dataclasses import dataclass, field
from datetime import datetime
import os
from typing import List, Optional

import cv2

from .exif_metadata import ExifMetadata
from ..timestamp  import toTimeStamp, fromTimeStamp


class SupportedMIME:
    FOURCC = {
        "video/mp4": cv2.VideoWriter_fourcc(*"mp4v"),
        "video/mov": cv2.VideoWriter_fourcc(*"mp4v"),
        "video/x-msvideo": cv2.VideoWriter_fourcc(*"MJPG"),
        "video/x-matroska": cv2.VideoWriter_fourcc(*"H264"),
    }
    MIME_TYPES = {
        "image/jpeg": {"extension": "jpg"},
        "image/png": {"extension": "png"},
        "image/tiff": {"extension": "tif"},
        "image/gif": {"extension": "gif"},
        "image/webp": {"extension": "webp"},
        "video/mp4": {"extension": "mp4"},
        "video/mov": {"extension": "mov"},
        "video/x-msvideo": {"extension": "avi"},
        "video/x-matroska": {"extension": "mkv"},
    }


@dataclass
class BaseMedia:
    out_dir: str
    MIMEType: str
    width: int
    height: int
    fileName: Optional[str] = None
    label: Optional[str] = None
    CreateDate: Optional[int] = None
    comments: List[str] = field(default_factory=list)
    

    @classmethod
    def from_dict(cls, data: dict):
        processed_data = data.copy()
        if "CreateDate" in processed_data and processed_data["CreateDate"] is not None:
            processed_data["CreateDate"] = fromTimeStamp(processed_data["CreateDate"])
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in processed_data.items() if k in valid_keys}
        return cls(**filtered_data)

    def to_dict(self) -> dict:
        """Converts the BaseMediaDescription instance to a dictionary for JSON serialization."""
        data = self.__dict__.copy()
        # Convert datetime object back to milliseconds since epoch if it was converted
        if isinstance(data.get("CreateDate"), datetime):
            data["CreateDate"] = toTimeStamp(data["CreateDate"])
        return data

    @property
    def media_info(self):
        if self.MIMEType not in SupportedMIME.MIME_TYPES:
            raise Exception(
                f"Error: Unsupported MIME type '{self.MIMEType}'. Supported types are: {list(SupportedMIME.MIME_TYPES.keys())}"
            )

        return SupportedMIME.MIME_TYPES[self.MIMEType]

    @property
    def fourcc_code(self):
        return SupportedMIME.FOURCC.get(self.MIMEType)

    @property
    def fileextension(self):
        return f".{self.media_info['extension']}"

    @property
    def filepath(self):
        path = os.path.join(
            self.out_dir, f"{self.fileName}{self.fileextension}"
        )
        directory, _ = os.path.split(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        return path

    @property
    def temp_filepath(self):
        path = os.path.join(
            self.out_dir, f"temp_{self.fileName}{self.fileextension}"
        )
        directory, _ = os.path.split(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        return path

    def generate(self):
        raise Exception("Implement in subclass")

    def update_metadata(self):
        if not os.path.exists(self.temp_filepath):
            raise Exception(f"Error: failed to read {self.temp_filepath}")

        ExifMetadata(
            MIMEType=self.MIMEType,
            CreateDate=self.CreateDate,
            UserComments=self.comments,
        ).write(self.temp_filepath)
