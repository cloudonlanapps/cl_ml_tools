from dataclasses import dataclass
import math
import random
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class Shape:
    thickness: int = 1
    color: Optional[Tuple[int, int, int]] = (0, 0, 0)

    @classmethod
    def from_dict(cls, data: dict):
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def draw(self, frame: np.ndarray):
        raise Exception("override on derived class")


class Circle(Shape):
    def draw(self, frame: np.ndarray):
        height, width, _ = frame.shape
        center = (random.randint(0, width), random.randint(0, height))
        radius = random.randint(10, min(width, height) // 8)
        cv2.circle(frame, center, radius, self.color, self.thickness)


class Rectangle(Shape):
    def draw(self, frame: np.ndarray):
        height, width, _ = frame.shape
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        pt1 = (min(x1, x2), min(y1, y2))
        pt2 = (max(x1, x2), max(y1, y2))
        cv2.rectangle(frame, pt1, pt2, self.color, self.thickness)


class Line(Shape):
    def draw(self, frame: np.ndarray):
        height, width, _ = frame.shape
        pt1 = (random.randint(0, width), random.randint(0, height))
        pt2 = (random.randint(0, width), random.randint(0, height))
        thickness = self.thickness if self.thickness > 0 else 1
        cv2.line(frame, pt1, pt2, self.color, thickness)


class Triangle(Shape):
    def draw(self, frame: np.ndarray):
        height, width, _ = frame.shape
        pt1 = (random.randint(0, width), random.randint(0, height))
        pt2 = (random.randint(0, width), random.randint(0, height))
        pt3 = (random.randint(0, width), random.randint(0, height))
        pts = np.array([[pt1, pt2, pt3]], np.int32)
        if self.thickness == -1:
            cv2.fillPoly(frame, pts, self.color)
        else:
            cv2.polylines(
                frame, pts, isClosed=True, color=self.color, thickness=self.thickness
            )


@dataclass
class AnimatedShape(Shape):
    is_initialized: Optional[bool] = False
    center: Optional[Tuple[int, int]] = None

    @classmethod
    def from_dict(cls, data: dict):
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_init_args = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_init_args)

    def initialize(self, width: int, height: int):
        raise Exception("override on derived class")

    def draw(self, frame: np.ndarray):
        raise Exception("override on derived class")


@dataclass
class BouncingCircle(AnimatedShape):
    radius: Optional[int] = None
    dx: Optional[int] = None
    dy: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict):
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_init_args = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_init_args)

    def initialize(self, width: int, height: int):
        self.radius = (
            random.randint(10, min(width, height) // 8)
            if self.radius is None
            else self.radius
        )
        self.center = (
            (random.randint(0, width), random.randint(0, height))
            if self.center is None
            else self.center
        )
        x, y = self.center
        radius = self.radius
        self.center = (
            max(radius, min(x, width - radius)),
            max(radius, min(y, height - radius)),
        )
        self.dx = random.choice([-5, 5]) if self.dx is None else self.dx
        self.dy = random.choice([-3, 3]) if self.dy is None else self.dy
        self.is_initialized = True

    def update(self, width: int, height: int):
        x, y = self.center
        x += self.dx
        y += self.dy        
        self.dx *= -1 if x + self.radius > width or x - self.radius < 0 else 1
        self.dy *= -1 if y + self.radius > height or y - self.radius < 0 else 1
        self.center = (x + self.dx, y + self.dy)

    def draw(self, frame: np.ndarray):
        height, width, _ = frame.shape
        if not self.is_initialized:
            self.initialize(width, height)
        else:
            self.update(width, height)
        cv2.circle(frame, self.center, self.radius, self.color, -1)


class MovingLine(AnimatedShape):
    length: Optional[int] = None
    angle_degrees: Optional[int] = None
    dx: Optional[int] = None
    dy: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict):
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_init_args = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_init_args)

    def initialize(self, width: int, height: int):
        self.center = (
            (random.randint(0, width), random.randint(0, height))
            if self.center is None
            else self.center
        )
        self.length = (
            random.randint(min(width, height) // 16, min(width, height) // 4)
            if self.length is None
            else self.length
        )
        self.angle_degrees = (
            random.randint(-90, 90)
            if self.angle_degrees is None
            else self.angle_degrees
        )
        self.dx = random.choice([-5, 5]) if self.dx is None else self.dx
        self.dy = random.choice([-3, 3]) if self.dy is None else self.dy
        self.update(width, height)
        self.is_initialized = True

    def update(self, width: int, height: int):
        angle_radians = np.deg2rad(self.angle_degrees)
        dx = (self.length / 2) * np.cos(angle_radians)
        dy = (self.length / 2) * np.sin(angle_radians)
        cx, cy = self.center
        x1 = cx - dx
        y1 = cy - dy
        x2 = cx + dx
        y2 = cy + dy
        self.dx *= -1 if (x1 > width or x1 < 0) or (x2 > width or x2 < 0) else 1
        self.dy *= -1 if (y1 > height or y1 < 0) or (y2 > height or y2 < 0) else 1

        x1 = np.clip(x1, 0, width - 1)
        y1 = np.clip(y1, 0, height - 1)
        x2 = np.clip(x2, 0, width - 1)
        y2 = np.clip(y2, 0, height - 1)
        self.pt0 = (int(round(x1)), int(round(y1)))
        self.pt1 = (int(round(x2)), int(round(y2)))
        self.center = (cx + self.dx, cy + self.dy)

    def draw(self, frame: np.ndarray):
        height, width, _ = frame.shape
        if not self.is_initialized:
            self.initialize(width, height)
        else:
            self.update(width, height)
        cv2.line(frame, self.pt0, self.pt1, self.color, max(self.thickness, 1))


class PulsatingTriangle(AnimatedShape):
    base_size: Optional[int] = None
    max_pulse: Optional[int] = None
    pulse_speed: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict):
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_init_args = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_init_args)

    def initialize(self, width: int, height: int):
        self.base_size = (
            random.randint(min(width, height) // 8, min(width, height) // 4)
            if self.base_size is None
            else self.base_size
        )
        self.max_pulse = (
            self.base_size // 2 if self.max_pulse is None else self.max_pulse
        )
        self.pulse_speed = 0.1 if self.pulse_speed is None else self.pulse_speed

        self.center = (
            (random.randint(0, width), random.randint(0, height))
            if self.center is None
            else self.center
        )
        self.frame_idx = 0
        self.update(width, height)
        self.is_initialized = True

    def update(self, width: int, height: int):
        self.current_size = self.base_size + self.max_pulse * math.sin(
            self.frame_idx * self.pulse_speed
        )
        h = self.current_size * math.sqrt(3) / 2
        x, y = self.center
        self.tri_pts = np.array(
            [
                (x, y - 2 * h / 3),
                (x - self.current_size / 2, y + h / 3),
                (x + self.current_size / 2, y + h / 3),
            ],
            np.int32,
        )
        self.frame_idx = self.frame_idx + 1

    def draw(self, frame: np.ndarray):
        height, width, _ = frame.shape
        if not self.is_initialized:
            self.initialize(width, height)
        else:
            self.update(width, height)
        cv2.fillPoly(frame, [self.tri_pts], self.color)


class RotatingSquare(AnimatedShape):
    size: Optional[int] = None
    angle: Optional[int] = None
    angular_speed: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict):
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_init_args = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_init_args)

    def initialize(self, width: int, height: int):
        self.center = (
            (random.randint(0, width), random.randint(0, height))
            if self.center is None
            else self.center
        )
        self.size = (
            random.randint(min(width, height) // 8, min(width, height) // 4)
            if self.size is None
            else self.size
        )
        self.angle = random.randint(0, 360) if self.angle is None else self.angle
        self.angle = max(0, min(self.angle, 359))
        self.angular_speed = (
            random.choice([-3, 3]) if self.angular_speed is None else self.angular_speed
        )
        self.update(width, height)
        self.is_initialized = True

    def update(self, width: int, height: int):
        x, y = self.center
        half_size = self.size // 2
        base_vertices = np.array(
            [
                [-half_size, -half_size],
                [half_size, -half_size],
                [half_size, half_size],
                [-half_size, half_size],
            ],
            dtype=np.float32,
        )
        rotation_matrix = cv2.getRotationMatrix2D((0, 0), self.angle, 1)
        rotated_vertices = (
            rotation_matrix
            @ np.array([base_vertices[:, 0], base_vertices[:, 1], np.ones(4)])
        )[:2].T
        rotated_vertices[:, 0] += x
        rotated_vertices[:, 1] += y
        self.square_pts = np.int32(rotated_vertices).reshape((-1, 1, 2))
        self.angle = (self.angle + self.angular_speed) % 360

    def draw(self, frame: np.ndarray):
        height, width, _ = frame.shape
        if not self.is_initialized:
            self.initialize(width, height)
        else:
            self.update(width, height)
        cv2.fillPoly(frame, [self.square_pts], self.color)


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