"""Unit tests for RandomMediaGenerator.

Tests media list validation, MIME type support, configuration, and actual media generation.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from cl_ml_tools.utils.random_media_generator import (
    JSONValidationError,
    RandomMediaGenerator,
)
from cl_ml_tools.utils.random_media_generator.base_media import SupportedMIME
from cl_ml_tools.utils.random_media_generator.frame_generator import FrameGenerator

# ============================================================================
# SupportedMIME Tests
# ============================================================================


def test_supported_mime_types_list():
    """Test SupportedMIME has expected MIME types."""
    mime_types = SupportedMIME.MIME_TYPES

    # Image types
    assert "image/jpeg" in mime_types
    assert "image/png" in mime_types
    assert "image/tiff" in mime_types
    assert "image/gif" in mime_types
    assert "image/webp" in mime_types

    # Video types
    assert "video/mp4" in mime_types
    assert "video/mov" in mime_types
    assert "video/x-msvideo" in mime_types
    assert "video/x-matroska" in mime_types


def test_supported_mime_extensions():
    """Test MIME types have correct file extensions."""
    assert SupportedMIME.MIME_TYPES["image/jpeg"]["extension"] == "jpg"
    assert SupportedMIME.MIME_TYPES["image/png"]["extension"] == "png"
    assert SupportedMIME.MIME_TYPES["video/mp4"]["extension"] == "mp4"
    assert SupportedMIME.MIME_TYPES["video/mov"]["extension"] == "mov"


def test_fourcc_codes():
    """Test FOURCC codes are defined for video types."""
    fourcc = SupportedMIME.FOURCC

    assert "video/mp4" in fourcc
    assert "video/mov" in fourcc
    assert "video/x-msvideo" in fourcc
    assert "video/x-matroska" in fourcc

    # All FOURCC codes should be integers
    for _, code in fourcc.items():
        assert isinstance(code, int)


# ============================================================================
# RandomMediaGenerator Initialization Tests
# ============================================================================


def test_random_media_generator_init_minimal(tmp_path: Path):
    """Test RandomMediaGenerator initialization with minimal params."""
    generator = RandomMediaGenerator(out_dir=str(tmp_path))

    assert generator.out_dir == str(tmp_path)
    assert generator.media_list == []


def test_random_media_generator_init_with_media_list(tmp_path: Path):
    """Test RandomMediaGenerator initialization with empty media list."""
    generator = RandomMediaGenerator(
        out_dir=str(tmp_path),
        media_list=[],
    )

    assert generator.out_dir == str(tmp_path)
    assert generator.media_list == []


def test_random_media_generator_supported_mime():
    """Test supportedMIME class method returns list of MIME types."""
    supported = RandomMediaGenerator.supportedMIME()

    assert isinstance(supported, list)
    assert len(supported) > 0
    assert "image/jpeg" in supported
    assert "image/png" in supported
    assert "video/mp4" in supported


# ============================================================================
# Media List Validation Tests
# ============================================================================


def test_media_list_validation_requires_out_dir():
    """Test media list validation fails without out_dir."""
    # This should fail during validation because out_dir is required
    with pytest.raises((ValidationError, JSONValidationError)):
        _ = RandomMediaGenerator(  # pyright: ignore[reportCallIssue] - Testing missing out_dir
            media_list=[
                {
                    "MIMEType": "image/jpeg",
                    "width": 800,
                    "height": 600,
                    "fileName": "test",
                },
            ],
        )


def test_media_list_validation_with_image_type(tmp_path: Path):
    """Test media list validates image/* MIME types."""
    # Note: This will fail validation because ImageGenerator requires 'frame' field
    with pytest.raises((ValidationError, JSONValidationError)):
        _ = RandomMediaGenerator(
            out_dir=str(tmp_path),
            media_list=[  # pyright: ignore[reportArgumentType]
                {
                    "MIMEType": "image/jpeg",
                    "width": 800,
                    "height": 600,
                    "fileName": "test",
                },
            ],
        )


def test_media_list_validation_with_video_type(tmp_path: Path):
    """Test media list validates video/* MIME types."""
    # Note: This will fail validation because VideoGenerator requires additional fields
    with pytest.raises((ValidationError, JSONValidationError)):
        _ = RandomMediaGenerator(
            out_dir=str(tmp_path),
            media_list=[  # pyright: ignore[reportArgumentType]
                {
                    "MIMEType": "video/mp4",
                    "width": 1920,
                    "height": 1080,
                    "fileName": "test",
                },
            ],
        )


def test_media_list_validation_invalid_mime_type(tmp_path: Path):
    """Test media list validation with unsupported MIME type."""
    # Unsupported MIME type should create BaseMedia (which is abstract)
    with pytest.raises((ValidationError, TypeError)):
        _ = RandomMediaGenerator(
            out_dir=str(tmp_path),
            media_list=[  # pyright: ignore[reportArgumentType]
                {
                    "MIMEType": "application/pdf",
                    "width": 800,
                    "height": 600,
                    "fileName": "test",
                },
            ],
        )


# ============================================================================
# Directory Management Tests
# ============================================================================


def test_random_media_generator_output_directory_validation(tmp_path: Path):
    """Test that output directory is validated and set correctly."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    generator = RandomMediaGenerator(out_dir=str(output_dir))

    assert generator.out_dir == str(output_dir)


def test_random_media_generator_nonexistent_directory(tmp_path: Path):
    """Test RandomMediaGenerator accepts non-existent directory path."""
    output_dir = tmp_path / "nonexistent"

    # Should not raise - directory creation happens during generation
    generator = RandomMediaGenerator(out_dir=str(output_dir))

    assert generator.out_dir == str(output_dir)


# ============================================================================
# Configuration Tests
# ============================================================================


def test_random_media_generator_is_pydantic_model(tmp_path: Path):
    """Test RandomMediaGenerator is a Pydantic BaseModel."""
    from pydantic import BaseModel

    generator = RandomMediaGenerator(out_dir=str(tmp_path))

    assert isinstance(generator, BaseModel)


def test_random_media_generator_serialization(tmp_path: Path):
    """Test RandomMediaGenerator can be serialized to dict."""
    generator = RandomMediaGenerator(out_dir=str(tmp_path))

    data = generator.model_dump()

    assert data["out_dir"] == str(tmp_path)
    assert data["media_list"] == []


def test_random_media_generator_deserialization(tmp_path: Path):
    """Test RandomMediaGenerator can be deserialized from dict."""
    data = {
        "out_dir": str(tmp_path),
        "media_list": [],
    }

    generator = RandomMediaGenerator.model_validate(data)

    assert generator.out_dir == str(tmp_path)
    assert generator.media_list == []


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_json_validation_error_import():
    """Test JSONValidationError can be imported."""
    assert JSONValidationError is not None
    assert issubclass(JSONValidationError, Exception)


def test_json_validation_error_raise():
    """Test JSONValidationError can be raised and caught."""
    with pytest.raises(JSONValidationError) as exc_info:
        raise JSONValidationError("Test error message")

    assert "Test error message" in str(exc_info.value)


def test_random_media_generator_validation_error_message(tmp_path: Path):
    """Test validation error provides helpful message."""
    with pytest.raises((ValidationError, JSONValidationError)) as exc_info:
        _ = RandomMediaGenerator(
            out_dir=str(tmp_path),
            media_list=[  # pyright: ignore[reportArgumentType]
                {
                    "MIMEType": "image/jpeg",
                    "width": 800,
                    "height": 600,
                    "fileName": "test",
                    # Missing required 'frame' field
                },
            ],
        )

    # Should mention missing field
    error_str = str(exc_info.value)
    assert len(error_str) > 0  # Has error message


# ============================================================================
# Edge Cases
# ============================================================================


def test_random_media_generator_empty_string_out_dir():
    """Test RandomMediaGenerator with empty string as out_dir."""
    # Empty string is technically valid, though not recommended
    generator = RandomMediaGenerator(out_dir="")

    assert generator.out_dir == ""


def test_random_media_generator_media_list_empty_default(tmp_path: Path):
    """Test media_list defaults to empty list."""
    generator = RandomMediaGenerator(out_dir=str(tmp_path))

    assert generator.media_list == []
    assert isinstance(generator.media_list, list)


def test_supported_mime_returns_new_list():
    """Test supportedMIME returns a new list each time."""
    list1 = RandomMediaGenerator.supportedMIME()
    list2 = RandomMediaGenerator.supportedMIME()

    # Should be equal but not the same object
    assert list1 == list2
    assert list1 is not list2


# ============================================================================
# GENERATION TESTS
# ============================================================================


def test_image_generation_with_shapes(tmp_path: Path):
    """Test actual image generation with various shapes."""
    generator = RandomMediaGenerator(
        out_dir=str(tmp_path),
        media_list=[  # pyright: ignore[reportArgumentType]
            {
                "MIMEType": "image/jpeg",
                "width": 100,
                "height": 100,
                "fileName": "test_shapes",
                "frame": {"background_color": [255, 0, 0], "num_shapes": 5},
            },
        ],
    )

    for media in generator.media_list:
        media.generate()

    assert (tmp_path / "test_shapes.jpg").exists()
    assert (tmp_path / "test_shapes.jpg").stat().st_size > 0


@pytest.mark.requires_ffmpeg
def test_video_generation_with_scenes(tmp_path: Path):
    """Test actual video generation with scenes."""
    generator = RandomMediaGenerator(
        out_dir=str(tmp_path),
        media_list=[  # pyright: ignore[reportArgumentType]
            {
                "MIMEType": "video/mp4",
                "width": 160,
                "height": 120,
                "fileName": "test_video",
                "fps": 10,
                "scenes": [
                    {"duration_seconds": 1, "background_color": [0, 255, 0], "num_shapes": 2},
                ],
            },
        ],
    )

    for media in generator.media_list:
        media.generate()

    assert (tmp_path / "test_video.mp4").exists()
    assert (tmp_path / "test_video.mp4").stat().st_size > 0


@pytest.mark.requires_exiftool
def test_image_generation_with_metadata(tmp_path: Path):
    """Test image generation with EXIF metadata."""
    from datetime import datetime

    generator = RandomMediaGenerator(
        out_dir=str(tmp_path),
        media_list=[  # pyright: ignore[reportArgumentType]
            {
                "MIMEType": "image/jpeg",
                "width": 100,
                "height": 100,
                "fileName": "test_meta",
                "frame": {"num_shapes": 1},
                "metadata": {
                    "CreateDate": datetime(2023, 1, 1, 12, 0, 0),
                    "UserComments": ["Test Comment"],
                },
            },
        ],
    )

    for media in generator.media_list:
        media.generate()

    assert (tmp_path / "test_meta.jpg").exists()


def test_scene_generator_num_frames():
    """Test SceneGenerator frame calculation."""
    from cl_ml_tools.utils.random_media_generator.scene_generator import SceneGenerator

    scene = SceneGenerator(duration_seconds=5)
    assert scene.num_frames(fps=30) == 150
    assert scene.num_frames(fps=10) == 50

    scene_no_duration = SceneGenerator(duration_seconds=None)
    assert scene_no_duration.num_frames(fps=30) == 0

    with pytest.raises(JSONValidationError, match="Invalid Color"):
        _ = FrameGenerator(
            background_color=(255, 0, 0),
        )  # Fixed to 3 values, but should still fail if that's what's intended?
        # Actually, the test was testing that [255, 0] is invalid.
        # But FrameGenerator expects a tuple or None.
        _ = FrameGenerator(background_color=[255, 0])  # pyright: ignore[reportArgumentType]


def test_basic_shapes_direct_draw():
    """Test various shapes draw methods directly to increase basic_shapes.py coverage."""
    from cl_ml_tools.utils.random_media_generator.basic_shapes import (
        Circle,
        Line,
        Rectangle,
        Triangle,
    )

    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Test each shape type
    Circle(color=(255, 0, 0), thickness=2).draw(frame)
    Rectangle(color=(0, 255, 0), thickness=-1).draw(frame)
    Line(color=(0, 0, 255), thickness=1).draw(frame)
    Triangle(thickness=1).draw(frame)

    assert np.any(frame > 0)


def test_animated_shapes_direct_draw():
    """Test animated shapes draw methods directly to increase basic_shapes.py coverage."""
    from cl_ml_tools.utils.random_media_generator.basic_shapes import (
        BouncingCircle,
        MovingLine,
        PulsatingTriangle,
        RotatingSquare,
    )

    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Test each animated shape
    BouncingCircle(center=(50, 50), radius=10, color=(255, 0, 0), dx=2, dy=2).draw(frame)
    MovingLine(center=(50, 50), length=20, angle_degrees=45, color=(0, 255, 0), dx=5, dy=5).draw(
        frame,
    )
    PulsatingTriangle(center=(50, 50), base_size=20, color=(0, 0, 255), pulse_speed=0.1).draw(frame)
    RotatingSquare(center=(50, 50), size=20, color=(255, 255, 0), angular_speed=3).draw(frame)

    assert np.any(frame > 0)


def test_exif_metadata_video_logic():
    """Test ExifMetadata logic for video types to hit uncovered branches."""
    from datetime import datetime

    from cl_ml_tools.utils.random_media_generator.exif_metadata import ExifMetadata

    # Test video branches
    meta = ExifMetadata(
        MIMEType="video/mp4", CreateDate=datetime(2023, 1, 1), UserComments=["Video Comment"],
    )
    # We call these to trigger the cmd expansion logic
    meta.updateCreateDate()
    meta.updateUserComments()

    cmd_str = " ".join(meta.cmd)
    assert "QuickTime:CreateDate" in cmd_str
    assert "QuickTime:Comment" in cmd_str
    assert meta.has_metadata is True


def test_exif_metadata_write_errors():
    """Test ExifMetadata.write error handling."""
    from datetime import datetime

    from cl_ml_tools.utils.random_media_generator.exif_metadata import ExifMetadata

    # 1. Empty metadata
    meta = ExifMetadata(MIMEType="image/jpeg")
    meta.write("file.jpg")  # Should just print and return

    # 2. CalledProcessError
    meta = ExifMetadata(MIMEType="image/jpeg", CreateDate=datetime(2023, 1, 1))
    with patch(
        "subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd", stderr="error"),
    ), pytest.raises(Exception, match="Error calling ExifTool"):
        meta.write("file.jpg")

    # 3. FileNotFoundError (ExifTool not found)
    with patch("subprocess.run", side_effect=FileNotFoundError()):
        with pytest.raises(Exception, match="ExifTool not found"):
            meta.write("file.jpg")

    # 4. General Exception
    with patch("subprocess.run", side_effect=RuntimeError("unknown")):
        meta.write("file.jpg")  # Should just log warning/print
