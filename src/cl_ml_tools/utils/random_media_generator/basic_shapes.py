import math
import random
from abc import ABCMeta, abstractmethod
from typing import ClassVar, TypeAlias, override

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

CVImage: TypeAlias = NDArray[np.uint8]


class Shape(BaseModel, metaclass=ABCMeta):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore",
        arbitrary_types_allowed=True,
    )

    thickness: int = 1
    color: tuple[int, int, int] = (0, 0, 0)

    @staticmethod
    def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(value, high))

    @abstractmethod
    def draw(self, _frame: CVImage) -> None: ...


class Circle(Shape):
    @override
    def draw(self, frame: CVImage) -> None:
        height, width, _ = frame.shape
        center = (random.randint(0, width), random.randint(0, height))
        radius = random.randint(10, min(width, height) // 8)
        _ = cv2.circle(frame, center, radius, self.color, self.thickness)


class Rectangle(Shape):
    @override
    def draw(self, frame: CVImage) -> None:
        height, width, _ = frame.shape
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        pt1 = (min(x1, x2), min(y1, y2))
        pt2 = (max(x1, x2), max(y1, y2))
        _ = cv2.rectangle(frame, pt1, pt2, self.color, self.thickness)


class Line(Shape):
    @override
    def draw(self, frame: CVImage) -> None:
        height, width, _ = frame.shape
        pt1 = (random.randint(0, width), random.randint(0, height))
        pt2 = (random.randint(0, width), random.randint(0, height))
        thickness = self.thickness if self.thickness > 0 else 1
        _ = cv2.line(frame, pt1, pt2, self.color, thickness)


class Triangle(Shape):
    @override
    def draw(self, frame: CVImage) -> None:
        height, width, _ = frame.shape
        pt1 = (random.randint(0, width), random.randint(0, height))
        pt2 = (random.randint(0, width), random.randint(0, height))
        pt3 = (random.randint(0, width), random.randint(0, height))
        pts = np.array([[pt1, pt2, pt3]], np.int32)
        if self.thickness == -1:
            _ = cv2.fillPoly(frame, [pts], self.color)
        else:
            _ = cv2.polylines(
                frame, [pts], isClosed=True, color=self.color, thickness=self.thickness
            )


class AnimatedShape(Shape, metaclass=ABCMeta):
    """
    Abstract base class for animated shapes.

    - Explicitly abstract (ABCMeta)
    - Safe with Pydantic BaseModel
    - basedpyright strict-mode compliant
    """

    # Optional: inherit this from Shape if already defined there
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore",
        arbitrary_types_allowed=True,
    )

    is_initialized: bool = False
    center: tuple[int, int] | None = None

    @abstractmethod
    def initialize(self, width: int, height: int) -> None:
        """Initialize animation state based on frame size."""
        ...


class BouncingCircle(AnimatedShape):
    radius: int | None = None
    dx: int | None = None
    dy: int | None = None
    center: tuple[int, int] | None = None

    @override
    def initialize(self, width: int, height: int) -> None:
        self.radius = self.radius or random.randint(10, min(width, height) // 8)
        self.center = self.center or (
            random.randint(0, width),
            random.randint(0, height),
        )

        x, y = self.center
        r = self.radius
        self.center = (
            max(r, min(x, width - r)),
            max(r, min(y, height - r)),
        )

        self.dx = self.dx or random.choice([-5, 5])
        self.dy = self.dy or random.choice([-3, 3])
        self.is_initialized: bool = True

    def update(self, width: int, height: int) -> None:
        if not self.center or not self.dx or not self.dy or not self.radius:
            raise ValueError("Not initialized")
        x, y = self.center
        x += self.dx
        y += self.dy

        if x + self.radius > width or x - self.radius < 0:
            self.dx *= -1
        if y + self.radius > height or y - self.radius < 0:
            self.dy *= -1

        self.center = (x + self.dx, y + self.dy)

    def _require_initialized(self) -> tuple[int, tuple[int, int]]:
        if self.radius is None or self.center is None or self.dx is None or self.dy is None:
            raise RuntimeError("BouncingCircle used before initialization")
        return self.radius, self.center

    @override
    def draw(self, frame: CVImage) -> None:
        height, width, _ = frame.shape
        if not self.is_initialized:
            self.initialize(width, height)
        else:
            self.update(width, height)
        radius, center = self._require_initialized()
        _ = cv2.circle(frame, center, radius, tuple(map(int, self.color)), -1)


class MovingLine(AnimatedShape):
    length: int | None = None
    angle_degrees: int | None = None
    dx: int | None = None
    dy: int | None = None
    center: tuple[int, int] | None = None

    pt0: tuple[int, int] | None = None
    pt1: tuple[int, int] | None = None

    @override
    def initialize(self, width: int, height: int) -> None:
        self.center = self.center or (
            random.randint(0, width),
            random.randint(0, height),
        )
        self.length = self.length or random.randint(
            min(width, height) // 16, min(width, height) // 4
        )
        self.angle_degrees = self.angle_degrees or random.randint(-90, 90)
        self.dx = self.dx or random.choice([-5, 5])
        self.dy = self.dy or random.choice([-3, 3])

        self.update(width, height)
        self.is_initialized: bool = True

    def update(self, width: int, height: int) -> None:
        # ---- explicit narrowing ----
        if (
            self.center is None
            or self.dx is None
            or self.dy is None
            or self.angle_degrees is None
            or self.length is None
        ):
            raise ValueError("Not initialized")

        center: tuple[int, int] = self.center
        dx_i: int = self.dx
        dy_i: int = self.dy
        angle_deg: int = self.angle_degrees
        length: int = self.length

        # ---- correct numeric type ----
        angle: float = math.radians(angle_deg)

        half_len: float = length / 2.0
        dx_f: float = half_len * math.cos(angle)
        dy_f: float = half_len * math.sin(angle)

        cx, cy = center
        x1, y1 = cx - dx_f, cy - dy_f
        x2, y2 = cx + dx_f, cy + dy_f

        if x1 < 0 or x2 > width:
            dx_i *= -1
        if y1 < 0 or y2 > height:
            dy_i *= -1

        x1 = Shape.clamp(x1, 0, width - 1)
        y1 = Shape.clamp(y1, 0, height - 1)
        x2 = Shape.clamp(x2, 0, width - 1)
        y2 = Shape.clamp(y2, 0, height - 1)

        self.pt0 = (int(round(x1)), int(round(y1)))
        self.pt1 = (int(round(x2)), int(round(y2)))
        self.center = (cx + dx_i, cy + dy_i)
        self.dx = dx_i
        self.dy = dy_i

    def _require_initialized(
        self,
    ) -> tuple[
        tuple[int, int],
        tuple[int, int],
        int,
    ]:
        """
        Ensure MovingLine is fully initialized and return required state.

        Returns:
            (pt0, pt1, thickness)
        """
        if self.pt0 is None or self.pt1 is None:
            raise RuntimeError("MovingLine used before initialization")

        thickness: int = self.thickness if self.thickness > 0 else 1

        return self.pt0, self.pt1, thickness

    @override
    def draw(self, frame: CVImage) -> None:
        height, width, _ = frame.shape
        if not self.is_initialized:
            self.initialize(width, height)
        else:
            self.update(width, height)
        pt0, pt1, thickness = self._require_initialized()

        _ = cv2.line(frame, pt0, pt1, self.color, max(thickness, 1))


class PulsatingTriangle(AnimatedShape):
    # ---- configurable fields ----
    base_size: int | None = None
    max_pulse: int | None = None
    pulse_speed: float | None = None
    center: tuple[int, int] | None = None

    # ---- runtime state ----
    frame_idx: int = 0
    tri_pts: NDArray[np.int32] | None = None

    @override
    def initialize(self, width: int, height: int) -> None:
        if self.base_size is None:
            self.base_size = random.randint(
                min(width, height) // 8,
                min(width, height) // 4,
            )

        if self.max_pulse is None:
            self.max_pulse = self.base_size // 2

        if self.pulse_speed is None:
            self.pulse_speed = 0.1

        if self.center is None:
            self.center = (
                random.randint(0, width),
                random.randint(0, height),
            )

        self.frame_idx = 0
        self.update(width, height)
        self.is_initialized: bool = True

    def update(self, _width: int, _height: int) -> None:
        if (
            self.base_size is None
            or self.max_pulse is None
            or self.pulse_speed is None
            or self.center is None
        ):
            raise RuntimeError("PulsatingTriangle update before initialization")

        # ---- scalar math: use math, not NumPy ----
        size: float = self.base_size + self.max_pulse * math.sin(self.frame_idx * self.pulse_speed)
        h: float = size * math.sqrt(3.0) / 2.0

        cx, cy = self.center

        self.tri_pts = np.array(
            [
                (cx, cy - 2.0 * h / 3.0),
                (cx - size / 2.0, cy + h / 3.0),
                (cx + size / 2.0, cy + h / 3.0),
            ],
            dtype=np.int32,
        )

        self.frame_idx += 1

    def _require_initialized(
        self,
    ) -> NDArray[np.int32]:
        if self.tri_pts is None:
            raise RuntimeError("PulsatingTriangle used before initialization")

        return self.tri_pts

    @override
    def draw(self, frame: CVImage) -> None:
        height, width, _ = frame.shape

        if not self.is_initialized:
            self.initialize(width, height)
        else:
            self.update(width, height)

        tri_pts = self._require_initialized()

        _ = cv2.fillPoly(frame, [tri_pts], self.color)


class RotatingSquare(AnimatedShape):
    size: int | None = None
    angle: int | None = None
    angular_speed: int | None = None
    center: tuple[int, int] | None = None

    square_pts: NDArray[np.int32] | None = None

    @override
    def initialize(self, width: int, height: int) -> None:
        if self.center is None:
            self.center = (
                random.randint(0, width),
                random.randint(0, height),
            )

        if self.size is None:
            self.size = random.randint(
                min(width, height) // 8,
                min(width, height) // 4,
            )

        if self.angle is None:
            self.angle = random.randint(0, 359)

        if self.angular_speed is None:
            self.angular_speed = random.choice([-3, 3])

        self.update(width, height)
        self.is_initialized: bool = True

    def update(self, _width: int, _height: int) -> None:
        if (
            self.center is None
            or self.size is None
            or self.angle is None
            or self.angular_speed is None
        ):
            raise RuntimeError("RotatingSquare update before initialization")

        cx, cy = self.center
        size: int = self.size
        angle_deg: int = self.angle
        angular_speed: int = self.angular_speed

        half: float = size / 2.0
        angle_rad: float = math.radians(angle_deg)

        base = np.array(
            [
                (-half, -half),
                (half, -half),
                (half, half),
                (-half, half),
            ],
            dtype=np.float32,
        )

        cos_a: float = math.cos(angle_rad)
        sin_a: float = math.sin(angle_rad)

        # ---- rotate manually (avoid NumPy scalar Any) ----
        rotated = np.empty_like(base)
        rotated[:, 0] = base[:, 0] * cos_a - base[:, 1] * sin_a
        rotated[:, 1] = base[:, 0] * sin_a + base[:, 1] * cos_a

        rotated[:, 0] += cx
        rotated[:, 1] += cy

        self.square_pts = rotated.astype(np.int32).reshape((-1, 1, 2))
        self.angle = (angle_deg + angular_speed) % 360

    def _require_square_pts(self) -> NDArray[np.int32]:
        if self.square_pts is None:
            raise RuntimeError("RotatingSquare used before initialization")
        return self.square_pts

    @override
    def draw(self, frame: CVImage) -> None:
        height, width, _ = frame.shape

        if not self.is_initialized:
            self.initialize(width, height)
        else:
            self.update(width, height)

        square_pts = self._require_square_pts()
        _ = cv2.fillPoly(frame, [square_pts], self.color)


Shapes = {
    "line": Line,
    "circle": Circle,
    "triangle": Triangle,
    "rectangle": Rectangle,
    "BouncingCircle": BouncingCircle,
    "MovingLine": MovingLine,
    "PulsatingTriangle": PulsatingTriangle,
    "RotatingSquare": RotatingSquare,
}
