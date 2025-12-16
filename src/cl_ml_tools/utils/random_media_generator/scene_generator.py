import random
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, field_validator, model_validator

from .basic_shapes import AnimatedShapes, Shape
from .frame_generator import FrameGenerator


class VideoWriterLike(Protocol):
    def write(self, frame: NDArray[np.uint8]) -> None: ...


class SceneGenerator(FrameGenerator):
    duration_seconds: int | None = None
    shapes: list[Shape] = Field(default_factory=list)

    @field_validator("duration_seconds")
    @classmethod
    def validate_duration(cls, v: int | None) -> int | None:
        if v is None:
            return None
        if v <= 0:
            raise ValueError("duration_seconds must be > 0")
        return v

    @model_validator(mode="after")
    def generate_animated_shapes(self):
        if self.num_shapes and self.num_shapes > 0:
            self.shapes = [
                AnimatedShapes[
                    random.choice(
                        [
                            "BouncingCircle",
                            "MovingLine",
                            "PulsatingTriangle",
                            "RotatingSquare",
                        ]
                    )
                ].model_validate(
                    {
                        "thickness": random.randint(-1, 3),
                        "color": (
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255),
                        ),
                    }
                )
                for _ in range(self.num_shapes)
            ]
        else:
            self.shapes = []

        return self

    # ---------------------------------
    # Frame math
    # ---------------------------------

    def num_frames(self, fps: int) -> int:
        if self.duration_seconds is None:
            return 0
        return self.duration_seconds * fps

    def render_to(
        self,
        *,
        out: object,
        fps: int,
        width: int,
        height: int,
    ) -> None:
        total_frames = self.num_frames(fps)
        if total_frames == 0:
            return

        for _ in range(total_frames):
            frame = self.generate_frame(width, height)
            out.write(frame)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
