from dataclasses import dataclass, field
from datetime import datetime
import random
from typing import List, Optional

import cv2

from .basic_shapes import Shapes, AnimatedShape
from .frame_generator import FrameGenerator


@dataclass
class SceneGenerator(FrameGenerator):
    duration: Optional[datetime] = None
    animated_shapes: Optional[List[AnimatedShape]] = field(default_factory=list)

    def with_shapes(self):
        if self.num_shapes and self.num_shapes > 0:
            self.shapes = [
                #
                Shapes[
                    random.choice(
                        [
                            "BouncingCircle",
                            "MovingLine",
                            "PulsatingTriangle",
                            "RotatingSquare",
                        ]
                    )
                ].from_dict(
                    {
                        "thickness": random.randint(-1, 3),
                        "color": (
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255),
                        ),
                    },
                )
                for _ in range(self.num_shapes)
            ]
        else:
            self.shapes = []
        return self

    @classmethod
    def from_dict(cls, data: dict):
        # Call parent's from_dict to handle inherited fields

        processed_data = super().from_dict(data)
        # Add specific fields for SceneDescription
        processed_data.duration = data.get("duration")

        return cls(**processed_data.__dict__).with_shapes()

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["duration"] = self.duration
        return data

    def num_frames(self, fps) -> int:
        return self.duration * fps

    def get_next_frame(self, width: int, height: int):
        frame = self.create_base_frame(width, height, self.background_color)

        if self.shapes:
            for shape in self.shapes:
                shape.draw(frame)

        """ for shape in self.animated_shapes:
            shape.update(index)
            shape.draw(frame) """

        return frame

    def generate(self, fps: int, out: cv2.VideoWriter, width: int, height: int):
        num_frames = self.num_frames(fps)
        if num_frames == 0:
            return

        for _ in range(num_frames):
            frame = self.get_next_frame(width, height)
            out.write(frame)
