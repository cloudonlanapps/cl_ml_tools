import random
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator, model_validator

from .basic_shapes import Shape, Shapes
from .errors import JSONValidationError

RawColor = Sequence[int] | tuple[int, int, int] | None


class FrameGenerator(BaseModel):
    background_color: tuple[int, int, int] | None = None  # BGR
    num_shapes: int | None = None
    shapes: list[Shape] = Field(default_factory=list)

    @field_validator("background_color", mode="before")
    @classmethod
    def validate_color(cls, v: RawColor):
        if v is None:
            return None

        if len(v) == 3:
            return tuple(v)

        raise JSONValidationError("Invalid Color [r, g, b]")

    @model_validator(mode="after")
    def generate_shapes(self):
        """
        Equivalent of with_shapes()
        """
        if self.num_shapes and self.num_shapes > 0:
            self.shapes = [
                Shapes[random.choice(["circle", "rectangle", "line", "triangle"])].model_validate(
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

    @staticmethod
    def create_base_frame(
        width: int, height: int, background_color: tuple[int, int, int] | None = None
    ) -> NDArray[np.uint8]:
        if background_color is None:
            background_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )

        return np.full((height, width, 3), background_color, dtype=np.uint8)

    def generate(self, width: int, height: int) -> NDArray[np.uint8]:
        frame = self.create_base_frame(width, height, self.background_color)

        for shape in self.shapes:
            shape.draw(frame)

        return frame
