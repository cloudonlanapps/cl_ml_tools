"""Unit tests for FFMPEGCommands in hls_streaming plugin."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from cl_ml_tools.plugins.hls_streaming.algo.ffmpeg_to_hls import (
    FFMPEGCommands,
    InternalServerError,
    NotFound,
)


@pytest.fixture
def ffmpeg_commands():
    """Fixture for FFMPEGCommands instance."""
    return FFMPEGCommands()


def test_to_hls_input_not_found(ffmpeg_commands, tmp_path):
    """Test to_hls raises NotFound if input file does NOT exist."""
    input_file = tmp_path / "nonexistent.mp4"
    output_dir = tmp_path / "output"

    with pytest.raises(NotFound, match="Input file does not exist"):
        ffmpeg_commands.to_hls(str(input_file), str(output_dir))


def test_to_hls_output_dir_not_found(ffmpeg_commands, tmp_path):
    """Test to_hls raises NotFound if output directory does NOT exist."""
    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"data")
    output_dir = tmp_path / "nonexistent_dir"

    with pytest.raises(NotFound, match="Output directory does not exist"):
        ffmpeg_commands.to_hls(str(input_file), str(output_dir))


def test_to_hls_success(ffmpeg_commands, tmp_path):
    """Test successful to_hls execution."""
    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"data")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    playlist_file = output_dir / "adaptive.m3u8"

    mock_result = MagicMock()
    mock_result.returncode = 0

    with patch("subprocess.run", return_value=mock_result):
        # In success case, ffmpeg would have created this file
        playlist_file.write_text("#EXTM3U")

        ffmpeg_commands.to_hls(str(input_file), str(output_dir))

        args, _ = subprocess.run.call_args
        command = args[0]
        assert "ffmpeg" in command
        assert str(input_file) in command
        assert "adaptive.m3u8" in command


def test_to_hls_ffmpeg_failure(ffmpeg_commands, tmp_path):
    """Test to_hls handles ffmpeg failure."""
    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"data")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "Detailed ffmpeg error"

    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(InternalServerError, match="FFmpeg command failed"):
            ffmpeg_commands.to_hls(str(input_file), str(output_dir))


def test_to_hls_subprocess_exception(ffmpeg_commands, tmp_path):
    """Test to_hls handles subprocess exception."""
    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"data")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch("subprocess.run", side_effect=OSError("Failed to start")):
        with pytest.raises(InternalServerError, match="Failed to start ffmpeg"):
            ffmpeg_commands.to_hls(str(input_file), str(output_dir))


def test_to_hls_playlist_not_created(ffmpeg_commands, tmp_path):
    """Test to_hls raises NotFound if playlist is not created by ffmpeg."""
    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"data")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = "" # Ensure it's a string to avoid TypeError in join

    with patch("subprocess.run", return_value=mock_result):
        # playlist_file is NOT created
        with pytest.raises(NotFound, match="Failed to create"):
            ffmpeg_commands.to_hls(str(input_file), str(output_dir))

