"""Comprehensive test suite for random media generator utility."""
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest

from cl_ml_tools.utils.random_media_generator import (
    JSONValidationError,
    RandomMediaGenerator,
)
from cl_ml_tools.utils.random_media_generator.base_media import SupportedMIME
from cl_ml_tools.utils.random_media_generator.exif_metadata import ExifMetadata
from cl_ml_tools.utils.random_media_generator.frame_generator import FrameGenerator
from cl_ml_tools.utils.random_media_generator.image_generator import ImageGenerator
from cl_ml_tools.utils.random_media_generator.scene_generator import SceneGenerator
from cl_ml_tools.utils.random_media_generator.video_generator import VideoGenerator

# ============================================================================
# Test Class 1: JSONValidationError
# ============================================================================


class TestJSONValidationError:
    """Test custom JSONValidationError exception."""

    def test_error_creation(self) -> None:
        """Test creating error with custom message."""
        error = JSONValidationError("Test error message")
        assert error.message == "Test error message"

    def test_error_default_message(self) -> None:
        """Test default error message."""
        error = JSONValidationError()
        assert error.message == "An unknown custom error occurred."

    def test_error_str_representation(self) -> None:
        """Test string representation of error."""
        error = JSONValidationError("Test error")
        assert str(error) == "CustomError: Test error"

    def test_error_is_exception(self) -> None:
        """Test that error is an Exception subclass."""
        error = JSONValidationError("Test")
        assert isinstance(error, Exception)


# ============================================================================
# Test Class 2: SupportedMIME
# ============================================================================


class TestSupportedMIME:
    """Test SupportedMIME utility class."""

    def test_supported_image_mime_types(self) -> None:
        """Test supported image MIME types."""
        assert "image/jpeg" in SupportedMIME.MIME_TYPES
        assert "image/png" in SupportedMIME.MIME_TYPES
        assert "image/tiff" in SupportedMIME.MIME_TYPES
        assert "image/gif" in SupportedMIME.MIME_TYPES
        assert "image/webp" in SupportedMIME.MIME_TYPES

    def test_supported_video_mime_types(self) -> None:
        """Test supported video MIME types."""
        assert "video/mp4" in SupportedMIME.MIME_TYPES
        assert "video/mov" in SupportedMIME.MIME_TYPES
        assert "video/x-msvideo" in SupportedMIME.MIME_TYPES
        assert "video/x-matroska" in SupportedMIME.MIME_TYPES

    def test_fourcc_codes(self) -> None:
        """Test FOURCC codes for video formats."""
        assert "video/mp4" in SupportedMIME.FOURCC
        assert "video/mov" in SupportedMIME.FOURCC
        assert "video/x-msvideo" in SupportedMIME.FOURCC
        assert "video/x-matroska" in SupportedMIME.FOURCC

    def test_mime_type_extensions(self) -> None:
        """Test that MIME types have extensions."""
        assert SupportedMIME.MIME_TYPES["image/jpeg"]["extension"] == "jpg"
        assert SupportedMIME.MIME_TYPES["image/png"]["extension"] == "png"
        assert SupportedMIME.MIME_TYPES["video/mp4"]["extension"] == "mp4"


# ============================================================================
# Test Class 3: ExifMetadata
# ============================================================================


class TestExifMetadata:
    """Test ExifMetadata class."""

    def test_exif_metadata_creation(self) -> None:
        """Test creating ExifMetadata with basic fields."""
        metadata = ExifMetadata(MIMEType="image/jpeg")
        assert metadata.MIMEType == "image/jpeg"
        assert metadata.CreateDate is None
        assert metadata.UserComments == []

    def test_exif_with_create_date(self) -> None:
        """Test ExifMetadata with CreateDate."""
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        metadata = ExifMetadata(MIMEType="image/jpeg", CreateDate=dt)
        assert metadata.CreateDate == dt

    def test_exif_with_user_comments(self) -> None:
        """Test ExifMetadata with UserComments."""
        metadata = ExifMetadata(
            MIMEType="image/jpeg", UserComments=["Comment 1", "Comment 2"]
        )
        assert len(metadata.UserComments) == 2
        assert "Comment 1" in metadata.UserComments

    def test_update_create_date_image(self) -> None:
        """Test updateCreateDate for image MIME type."""
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        metadata = ExifMetadata(MIMEType="image/jpeg", CreateDate=dt)
        metadata.updateCreateDate()

        assert metadata.has_metadata is True
        assert any("-DateTimeOriginal=" in arg for arg in metadata.cmd)

    def test_update_create_date_video(self) -> None:
        """Test updateCreateDate for video MIME type."""
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        metadata = ExifMetadata(MIMEType="video/mp4", CreateDate=dt)
        metadata.updateCreateDate()

        assert metadata.has_metadata is True
        assert any("-QuickTime:CreateDate=" in arg for arg in metadata.cmd)

    def test_update_user_comments(self) -> None:
        """Test updateUserComments."""
        metadata = ExifMetadata(MIMEType="image/jpeg", UserComments=["Test comment"])
        metadata.updateUserComments()

        assert metadata.has_metadata is True
        assert any("-UserComment=Test comment" in arg for arg in metadata.cmd)

    @patch("subprocess.run")
    def test_write_metadata_success(self, mock_run: Mock) -> None:
        """Test writing metadata successfully."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        metadata = ExifMetadata(MIMEType="image/jpeg", CreateDate=dt)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Mock file existence
            with patch("os.path.exists", return_value=True):
                metadata.write(tmp_path)

            mock_run.assert_called_once()
            assert tmp_path in mock_run.call_args[0][0]
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_write_empty_metadata(self, capsys) -> None:
        """Test writing empty metadata prints message."""
        metadata = ExifMetadata(MIMEType="image/jpeg")

        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            metadata.write(tmp.name)

        captured = capsys.readouterr()
        assert "Metadata is empty" in captured.out


# ============================================================================
# Test Class 4: FrameGenerator
# ============================================================================


class TestFrameGenerator:
    """Test FrameGenerator class."""

    def test_frame_generator_creation(self) -> None:
        """Test creating FrameGenerator with defaults."""
        frame_gen = FrameGenerator()
        assert frame_gen.background_color is None
        assert frame_gen.num_shapes is None
        assert frame_gen.shapes == []

    def test_frame_generator_with_background_color(self) -> None:
        """Test FrameGenerator with background color."""
        frame_gen = FrameGenerator(background_color=(255, 0, 0))
        assert frame_gen.background_color == (255, 0, 0)

    def test_frame_generator_color_validation(self) -> None:
        """Test color validation."""
        frame_gen = FrameGenerator(background_color=[100, 150, 200])
        assert frame_gen.background_color == (100, 150, 200)

    def test_frame_generator_invalid_color(self) -> None:
        """Test invalid color raises error."""
        with pytest.raises(JSONValidationError, match="Invalid Color"):
            FrameGenerator(background_color=[100, 150])

    def test_frame_generator_with_num_shapes(self) -> None:
        """Test generating random shapes."""
        frame_gen = FrameGenerator(num_shapes=5)
        assert len(frame_gen.shapes) == 5

    def test_create_base_frame_with_color(self) -> None:
        """Test creating base frame with specified color."""
        frame = FrameGenerator.create_base_frame(
            width=100, height=100, background_color=(255, 0, 0)
        )
        assert frame.shape == (100, 100, 3)
        assert frame.dtype == np.uint8
        # Check that all pixels are blue (BGR format)
        assert np.all(frame[:, :, 0] == 255)

    def test_create_base_frame_random_color(self) -> None:
        """Test creating base frame with random color."""
        frame = FrameGenerator.create_base_frame(width=50, height=50)
        assert frame.shape == (50, 50, 3)
        assert frame.dtype == np.uint8

    def test_generate_frame(self) -> None:
        """Test generating a frame."""
        frame_gen = FrameGenerator(background_color=(0, 0, 0), num_shapes=2)
        frame = frame_gen.generate_frame(width=100, height=100)

        assert frame.shape == (100, 100, 3)
        assert frame.dtype == np.uint8


# ============================================================================
# Test Class 5: SceneGenerator
# ============================================================================


class TestSceneGenerator:
    """Test SceneGenerator class."""

    def test_scene_generator_creation(self) -> None:
        """Test creating SceneGenerator."""
        scene = SceneGenerator(duration_seconds=5)
        assert scene.duration_seconds == 5

    def test_scene_duration_validation(self) -> None:
        """Test duration validation."""
        with pytest.raises(ValueError, match="duration_seconds must be > 0"):
            SceneGenerator(duration_seconds=-1)

    def test_num_frames_calculation(self) -> None:
        """Test calculating number of frames."""
        scene = SceneGenerator(duration_seconds=2)
        assert scene.num_frames(fps=30) == 60

    def test_num_frames_none_duration(self) -> None:
        """Test num_frames with None duration."""
        scene = SceneGenerator()
        assert scene.num_frames(fps=30) == 0

    def test_scene_with_animated_shapes(self) -> None:
        """Test generating animated shapes."""
        scene = SceneGenerator(num_shapes=3, duration_seconds=1)
        assert len(scene.shapes) == 3

    def test_render_to_video_writer(self) -> None:
        """Test rendering scene to video writer."""
        scene = SceneGenerator(
            background_color=(0, 0, 0), num_shapes=1, duration_seconds=1
        )

        mock_writer = Mock()
        scene.render_to(out=mock_writer, fps=30, width=100, height=100)

        # Should write 30 frames (1 second at 30 fps)
        assert mock_writer.write.call_count == 30

    def test_render_to_zero_duration(self) -> None:
        """Test rendering with zero duration."""
        scene = SceneGenerator(duration_seconds=None)
        mock_writer = Mock()

        scene.render_to(out=mock_writer, fps=30, width=100, height=100)

        # Should not write any frames
        mock_writer.write.assert_not_called()


# ============================================================================
# Test Class 6: ImageGenerator
# ============================================================================


class TestImageGenerator:
    """Test ImageGenerator class."""

    def test_image_generator_creation(self) -> None:
        """Test creating ImageGenerator with frame."""
        frame = FrameGenerator(background_color=(255, 255, 255))
        img_gen = ImageGenerator(
            out_dir="/tmp",
            MIMEType="image/jpeg",
            width=640,
            height=480,
            fileName="test",
            frame=frame,
        )

        assert img_gen.width == 640
        assert img_gen.height == 480
        assert img_gen.frame is not None

    def test_image_generator_frame_validation(self) -> None:
        """Test frame field validation from dict."""
        img_gen = ImageGenerator(
            out_dir="/tmp",
            MIMEType="image/jpeg",
            width=640,
            height=480,
            fileName="test",
            frame={"background_color": [255, 0, 0]},
        )

        assert isinstance(img_gen.frame, FrameGenerator)

    def test_image_generator_missing_frame(self) -> None:
        """Test that missing frame raises error."""
        with pytest.raises(JSONValidationError, match="missing 'frame' data"):
            ImageGenerator(
                out_dir="/tmp",
                MIMEType="image/jpeg",
                width=640,
                height=480,
                fileName="test",
            )

    @patch("subprocess.run")
    @patch("os.path.exists", return_value=True)
    @patch("os.rename")
    @patch("cv2.imwrite")
    def test_image_generate(
        self,
        mock_imwrite: Mock,
        mock_rename: Mock,
        mock_exists: Mock,
        mock_subprocess: Mock,
    ) -> None:
        """Test image generation."""
        mock_imwrite.return_value = True
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        frame = FrameGenerator(background_color=(0, 0, 0))
        img_gen = ImageGenerator(
            out_dir="/tmp",
            MIMEType="image/jpeg",
            width=100,
            height=100,
            fileName="test",
            frame=frame,
        )

        img_gen.generate()

        mock_imwrite.assert_called_once()
        mock_rename.assert_called_once()


# ============================================================================
# Test Class 7: VideoGenerator
# ============================================================================


class TestVideoGenerator:
    """Test VideoGenerator class."""

    def test_video_generator_creation(self) -> None:
        """Test creating VideoGenerator."""
        scene = SceneGenerator(duration_seconds=1)
        vid_gen = VideoGenerator(
            out_dir="/tmp",
            MIMEType="video/mp4",
            width=640,
            height=480,
            fileName="test",
            scenes=[scene],
            fps=30,
        )

        assert vid_gen.fps == 30
        assert len(vid_gen.scenes) == 1

    def test_video_generator_fps_validation(self) -> None:
        """Test FPS validation."""
        with pytest.raises(ValueError, match="fps must be > 0"):
            VideoGenerator(
                out_dir="/tmp",
                MIMEType="video/mp4",
                width=640,
                height=480,
                fileName="test",
                scenes=[],
                fps=0,
            )

    def test_video_generator_scenes_validation(self) -> None:
        """Test scenes field validation from dicts."""
        vid_gen = VideoGenerator(
            out_dir="/tmp",
            MIMEType="video/mp4",
            width=640,
            height=480,
            fileName="test",
            scenes=[{"duration_seconds": 2}],
            fps=30,
        )

        assert len(vid_gen.scenes) == 1
        assert isinstance(vid_gen.scenes[0], SceneGenerator)

    def test_video_generator_missing_scenes(self) -> None:
        """Test that missing scenes raises error."""
        with pytest.raises(JSONValidationError, match="'scenes' not found"):
            VideoGenerator(
                out_dir="/tmp",
                MIMEType="video/mp4",
                width=640,
                height=480,
                fileName="test",
                fps=30,
            )

    @patch("subprocess.run")
    @patch("os.path.exists", return_value=True)
    @patch("os.rename")
    @patch("cv2.VideoWriter")
    def test_video_generate(
        self,
        mock_writer_class: Mock,
        mock_rename: Mock,
        mock_exists: Mock,
        mock_subprocess: Mock,
    ) -> None:
        """Test video generation."""
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        mock_writer_class.return_value = mock_writer
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        scene = SceneGenerator(
            background_color=(0, 0, 0), num_shapes=1, duration_seconds=1
        )
        vid_gen = VideoGenerator(
            out_dir="/tmp",
            MIMEType="video/mp4",
            width=100,
            height=100,
            fileName="test",
            scenes=[scene],
            fps=30,
        )

        vid_gen.generate()

        mock_writer.release.assert_called_once()
        mock_rename.assert_called_once()


# ============================================================================
# Test Class 8: BaseMedia
# ============================================================================


class TestBaseMedia:
    """Test BaseMedia abstract class."""

    def test_base_media_properties(self) -> None:
        """Test BaseMedia properties."""
        # Create concrete subclass for testing
        scene = SceneGenerator(duration_seconds=1)
        media = VideoGenerator(
            out_dir="/tmp",
            MIMEType="video/mp4",
            width=640,
            height=480,
            fileName="test_video",
            scenes=[scene],
        )

        assert media.fileextension == ".mp4"
        assert media.fourcc_code == cv2.VideoWriter.fourcc(*"mp4v")

    def test_base_media_unsupported_mime(self) -> None:
        """Test unsupported MIME type raises error."""
        scene = SceneGenerator(duration_seconds=1)
        media = VideoGenerator(
            out_dir="/tmp",
            MIMEType="video/unsupported",
            width=640,
            height=480,
            fileName="test",
            scenes=[scene],
        )

        with pytest.raises(Exception, match="Unsupported MIME type"):
            _ = media.media_info

    def test_base_media_create_date_validation(self) -> None:
        """Test CreateDate timestamp validation."""
        scene = SceneGenerator(duration_seconds=1)
        media = VideoGenerator(
            out_dir="/tmp",
            MIMEType="video/mp4",
            width=640,
            height=480,
            fileName="test",
            scenes=[scene],
            CreateDate=1704067200000,  # Timestamp
        )

        assert isinstance(media.CreateDate, datetime)
        assert media.CreateDate.year == 2024

    def test_base_media_filepath_creates_directory(self) -> None:
        """Test that filepath property creates directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = SceneGenerator(duration_seconds=1)
            media = VideoGenerator(
                out_dir=f"{tmp_dir}/subdir",
                MIMEType="video/mp4",
                width=640,
                height=480,
                fileName="test",
                scenes=[scene],
            )

            filepath = media.filepath
            assert Path(filepath).parent.exists()


# ============================================================================
# Test Class 9: RandomMediaGenerator
# ============================================================================


class TestRandomMediaGenerator:
    """Test RandomMediaGenerator main class."""

    def test_random_media_generator_creation(self) -> None:
        """Test creating RandomMediaGenerator."""
        gen = RandomMediaGenerator(out_dir="/tmp", media_list=[])
        assert gen.out_dir == "/tmp"
        assert gen.media_list == []

    def test_supported_mime(self) -> None:
        """Test supportedMIME class method."""
        supported = RandomMediaGenerator.supportedMIME()
        assert "image/jpeg" in supported
        assert "video/mp4" in supported

    def test_media_list_validation_image(self) -> None:
        """Test media_list validation for images."""
        gen = RandomMediaGenerator(
            out_dir="/tmp",
            media_list=[
                {
                    "MIMEType": "image/jpeg",
                    "width": 640,
                    "height": 480,
                    "fileName": "test_img",
                    "frame": {"background_color": [255, 0, 0]},
                }
            ],
        )

        assert len(gen.media_list) == 1
        assert isinstance(gen.media_list[0], ImageGenerator)

    def test_media_list_validation_video(self) -> None:
        """Test media_list validation for videos."""
        gen = RandomMediaGenerator(
            out_dir="/tmp",
            media_list=[
                {
                    "MIMEType": "video/mp4",
                    "width": 640,
                    "height": 480,
                    "fileName": "test_vid",
                    "scenes": [{"duration_seconds": 2}],
                    "fps": 30,
                }
            ],
        )

        assert len(gen.media_list) == 1
        assert isinstance(gen.media_list[0], VideoGenerator)

    def test_media_list_validation_mixed(self) -> None:
        """Test media_list validation with mixed media types."""
        gen = RandomMediaGenerator(
            out_dir="/tmp",
            media_list=[
                {
                    "MIMEType": "image/jpeg",
                    "width": 640,
                    "height": 480,
                    "fileName": "test_img",
                    "frame": {"background_color": [255, 0, 0]},
                },
                {
                    "MIMEType": "video/mp4",
                    "width": 640,
                    "height": 480,
                    "fileName": "test_vid",
                    "scenes": [{"duration_seconds": 2}],
                    "fps": 30,
                },
            ],
        )

        assert len(gen.media_list) == 2
        assert isinstance(gen.media_list[0], ImageGenerator)
        assert isinstance(gen.media_list[1], VideoGenerator)

    def test_media_list_missing_out_dir(self) -> None:
        """Test that missing out_dir raises error."""
        with pytest.raises(JSONValidationError, match="'out_dir' must be provided"):
            # This should fail validation since out_dir is required before media_list
            RandomMediaGenerator(
                out_dir=None,  # type: ignore
                media_list=[{"MIMEType": "image/jpeg"}],
            )


# ============================================================================
# Test Class 10: Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    @patch("subprocess.run")
    @patch("os.path.exists", return_value=True)
    @patch("os.rename")
    @patch("cv2.imwrite")
    def test_generate_image_workflow(
        self,
        mock_imwrite: Mock,
        mock_rename: Mock,
        mock_exists: Mock,
        mock_subprocess: Mock,
    ) -> None:
        """Test complete image generation workflow."""
        mock_imwrite.return_value = True
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp_dir:
            gen = RandomMediaGenerator(
                out_dir=tmp_dir,
                media_list=[
                    {
                        "MIMEType": "image/jpeg",
                        "width": 100,
                        "height": 100,
                        "fileName": "test_image",
                        "frame": {"background_color": [255, 255, 255], "num_shapes": 3},
                    }
                ],
            )

            img = gen.media_list[0]
            img.generate()

            mock_imwrite.assert_called_once()

    @patch("subprocess.run")
    @patch("os.path.exists", return_value=True)
    @patch("os.rename")
    @patch("cv2.VideoWriter")
    def test_generate_video_workflow(
        self,
        mock_writer_class: Mock,
        mock_rename: Mock,
        mock_exists: Mock,
        mock_subprocess: Mock,
    ) -> None:
        """Test complete video generation workflow."""
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        mock_writer_class.return_value = mock_writer
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp_dir:
            gen = RandomMediaGenerator(
                out_dir=tmp_dir,
                media_list=[
                    {
                        "MIMEType": "video/mp4",
                        "width": 100,
                        "height": 100,
                        "fileName": "test_video",
                        "scenes": [
                            {
                                "background_color": [0, 0, 0],
                                "num_shapes": 2,
                                "duration_seconds": 1,
                            }
                        ],
                        "fps": 30,
                    }
                ],
            )

            vid = gen.media_list[0]
            vid.generate()

            mock_writer.release.assert_called_once()
            # 1 second at 30 fps = 30 frames
            assert mock_writer.write.call_count == 30
