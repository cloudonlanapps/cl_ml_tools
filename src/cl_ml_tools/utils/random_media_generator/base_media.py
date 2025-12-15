from __future__ import annotations

import os
from abc import ABCMeta, abstractmethod
from datetime import datetime

import cv2
from pydantic import BaseModel, field_serializer, field_validator

from ..timestamp import fromTimeStamp, toTimeStamp
from .exif_metadata import ExifMetadata


class SupportedMIME:
    FOURCC: dict[str, int] = {
        "video/mp4": cv2.VideoWriter.fourcc(*"mp4v"),
        "video/mov": cv2.VideoWriter.fourcc(*"mp4v"),
        "video/x-msvideo": cv2.VideoWriter.fourcc(*"MJPG"),
        "video/x-matroska": cv2.VideoWriter.fourcc(*"H264"),
    }
    MIME_TYPES: dict[str, dict[str, str]] = {
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


class BaseMedia(BaseModel, metaclass=ABCMeta):
    out_dir: str
    MIMEType: str
    width: int
    height: int
    fileName: str | None = None
    label: str | None = None
    CreateDate: datetime | None = None
    comments: list[str] = []

    @field_validator("CreateDate", mode="before")
    @classmethod
    def validate_create_date(cls, value: int | None):
        if value is None:
            return None

        return fromTimeStamp(value)

    @field_serializer("CreateDate")
    def serialize_create_date(self, value: datetime | None) -> int | None:
        if value is None:
            return None
        return toTimeStamp(value)

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
        path = os.path.join(self.out_dir, f"{self.fileName}{self.fileextension}")
        directory, _ = os.path.split(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        return path

    @property
    def temp_filepath(self):
        path = os.path.join(self.out_dir, f"temp_{self.fileName}{self.fileextension}")
        directory, _ = os.path.split(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        return path

    @abstractmethod
    def generate(self) -> None: ...

    def update_metadata(self):
        if not os.path.exists(self.temp_filepath):
            raise Exception(f"Error: failed to read {self.temp_filepath}")

        ExifMetadata(
            MIMEType=self.MIMEType,
            CreateDate=self.CreateDate,
            UserComments=self.comments,
        ).write(self.temp_filepath)
