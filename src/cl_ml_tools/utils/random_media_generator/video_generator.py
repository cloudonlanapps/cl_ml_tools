import os
from collections.abc import Mapping, Sequence
from typing import Protocol, override

import cv2
from pydantic import Field, field_validator, model_validator

from .base_media import BaseMedia
from .errors import JSONValidationError
from .scene_generator import SceneGenerator


class VideoWriterLike(Protocol):
    def write(self, frame: object) -> None: ...
    def release(self) -> None: ...
    def isOpened(self) -> bool: ...


RawScene = Mapping[str, object] | SceneGenerator
RawScenes = Sequence[RawScene]


class VideoGenerator(BaseMedia):
    scenes: list[SceneGenerator] = Field(default_factory=list)
    fps: int = 30

    @field_validator("fps")
    @classmethod
    def validate_fps(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("fps must be > 0")
        return v

    @field_validator("scenes", mode="before")
    @classmethod
    def validate_scenes(cls, v: RawScenes) -> list[SceneGenerator]:
        scenes: list[SceneGenerator] = []

        for item in v:
            if isinstance(item, SceneGenerator):
                scenes.append(item)
            else:
                scenes.append(SceneGenerator.model_validate(item))

        return scenes

    @model_validator(mode="after")
    def ensure_scenes_present(self):
        if not self.scenes:
            raise JSONValidationError("'scenes' not found for VideoDescription.")
        return self

    # ----------------------------
    # Video generation
    # ----------------------------

    def _create_writer(self) -> object:
        if not self.fourcc_code:
            raise ValueError("fourcc_code not defined")

        writer = cv2.VideoWriter(
            self.temp_filepath,
            self.fourcc_code,
            self.fps,
            (self.width, self.height),
        )

        if not writer.isOpened():
            raise Exception(f"Error: Could not open video writer for {self.filepath}")

        return writer

    @override
    def generate(self) -> None:
        if not self.fourcc_code:
            raise ValueError("fourcc_code not defined")

        writer = cv2.VideoWriter(
            self.temp_filepath,
            self.fourcc_code,
            self.fps,
            (self.width, self.height),
        )

        try:
            for scene in self.scenes:
                scene.render_to(
                    out=writer,
                    fps=self.fps,
                    width=self.width,
                    height=self.height,
                )
        finally:
            writer.release()

        print(f"Video '{self.fileName}' created by OpenCV.")

        self.update_metadata()
        os.rename(self.temp_filepath, self.filepath)
