from dataclasses import dataclass, field
import random
from typing import List, Optional, Tuple
import numpy as np

from .basic_shapes import Shape, Shapes
from .errors import JSONValidationError

@dataclass
class FrameGenerator:
    background_color: Optional[Tuple[int, int, int]] = None  # BGR
    num_shapes: Optional[int] = None
    shapes: Optional[int] = None
    shapes: List[Shape] = field(default_factory=list)

    @staticmethod
    def _convert_color_tuple(
        color_list: Optional[List[int]],
    ) -> Optional[Tuple[int, int, int]]:
        if color_list is None:
            return None
        if (
            isinstance(color_list, list)
            and len(color_list) == 3
            and all(isinstance(c, int) for c in color_list)
        ):
            return tuple(color_list)
        elif (
            isinstance(color_list, tuple)
            and len(color_list) == 3
            and all(isinstance(c, int) for c in color_list)
        ):
            return color_list

        raise JSONValidationError("Invalid Color [r, g, b]")

    @staticmethod
    def _convert_position_tuple(
        pos_list: Optional[List[int]],
    ) -> Optional[Tuple[int, int]]:
        if (
            isinstance(pos_list, list)
            and len(pos_list) == 2
            and all(isinstance(p, int) for p in pos_list)
        ):
            return tuple(pos_list)
        if pos_list is None:
            return None
        raise JSONValidationError("Invalid Position [x,y]")

    
    def with_shapes(self):
        if self.num_shapes and self.num_shapes > 0:
            self.shapes = [
                Shapes[random.choice(["circle", "rectangle", "line", "triangle"])].from_dict(
                    {"thickness":random.randint(-1, 3),
                    "color":(
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    )},
                )
                for _ in range(self.num_shapes)
            ]
        else:
            self.shapes = []
        return self

    @classmethod
    def from_dict(cls, data: dict):
        processed_data = data.copy()
        if "background_color" in processed_data:
            processed_data["background_color"] = cls._convert_color_tuple(
                processed_data.get("background_color")
            )
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in processed_data.items() if k in valid_keys}
        frameGenerator = cls(**filtered_data).with_shapes()
        return frameGenerator

    def to_dict(self) -> dict:
        """Converts the FrameDescription instance to a dictionary for JSON serialization."""
        return self.__dict__.copy()

    @staticmethod
    def create_base_frame(
        width: int, height: int, background_color: tuple = None
    ) -> np.ndarray:
        """Creates a blank frame with a specified or random background color."""
        if background_color is None:
            background_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
        return np.full((height, width, 3), background_color, dtype=np.uint8)

    def generate(self, width: int, height: int):
        frame = self.create_base_frame(width, height, self.background_color)
        if self.shapes:
            for shape in self.shapes:
                shape.draw(frame)

        return frame
