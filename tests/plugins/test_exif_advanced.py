"""Advanced unit tests for ExifToolWrapper.

Targets error handling and edge cases in MetadataExtractor to reach high coverage.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from cl_ml_tools.plugins.exif.algo.exif_tool_wrapper import MetadataExtractor


def test_metadata_extractor_init_fail():
    """Test MetadataExtractor initialization failure when exiftool is missing."""
    with patch.object(MetadataExtractor, "is_exiftool_available", return_value=False):
        with pytest.raises(RuntimeError, match="ExifTool is not installed"):
            MetadataExtractor()


@pytest.mark.parametrize(
    "exception",
    [
        FileNotFoundError,
        subprocess.CalledProcessError(1, ["exiftool"]),
        subprocess.TimeoutExpired(["exiftool"], 5),
    ],
)
def test_is_exiftool_available_exceptions(exception):
    """Test is_exiftool_available handles various exceptions."""
    with patch("subprocess.run", side_effect=exception):
        extractor = MetadataExtractor.__new__(MetadataExtractor)
        assert extractor.is_exiftool_available() is False


def test_extract_metadata_file_not_found():
    """Test extract_metadata raises FileNotFoundError for missing files."""
    with patch.object(MetadataExtractor, "is_exiftool_available", return_value=True):
        extractor = MetadataExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract_metadata("non_existent_file.jpg", ["Artist"])


def test_extract_metadata_empty_tags():
    """Test extract_metadata returns empty dict for empty tags."""
    with patch.object(MetadataExtractor, "is_exiftool_available", return_value=True):
        extractor = MetadataExtractor()
        with patch("pathlib.Path.exists", return_value=True):
            assert extractor.extract_metadata("file.jpg", []) == {}


def test_extract_metadata_called_process_error():
    """Test extract_metadata handles subprocess failure."""
    with patch.object(MetadataExtractor, "is_exiftool_available", return_value=True):
        extractor = MetadataExtractor()
        with patch("pathlib.Path.exists", return_value=True), patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "cmd", stderr="error"),
        ):
            assert extractor.extract_metadata("file.jpg", ["Artist"]) == {}


def test_extract_metadata_timeout():
    """Test extract_metadata handles timeout."""
    with patch.object(MetadataExtractor, "is_exiftool_available", return_value=True):
        extractor = MetadataExtractor()
        with patch("pathlib.Path.exists", return_value=True):
            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)):
                assert extractor.extract_metadata("file.jpg", ["Artist"]) == {}


def test_extract_metadata_json_error():
    """Test extract_metadata handles invalid JSON output."""
    with patch.object(MetadataExtractor, "is_exiftool_available", return_value=True):
        extractor = MetadataExtractor()
        with patch("pathlib.Path.exists", return_value=True):
            mock_result = MagicMock()
            mock_result.stdout = "invalid json"
            with patch("subprocess.run", return_value=mock_result):
                assert extractor.extract_metadata("file.jpg", ["Artist"]) == {}


def test_extract_metadata_empty_result():
    """Test extract_metadata handles empty JSON list."""
    with patch.object(MetadataExtractor, "is_exiftool_available", return_value=True):
        extractor = MetadataExtractor()
        with patch("pathlib.Path.exists", return_value=True):
            mock_result = MagicMock()
            mock_result.stdout = "[]"
            with patch("subprocess.run", return_value=mock_result):
                assert extractor.extract_metadata("file.jpg", ["Artist"]) == {}


def test_extract_metadata_all_exceptions():
    """Test extract_metadata_all handles various exceptions."""
    with patch.object(MetadataExtractor, "is_exiftool_available", return_value=True):
        extractor = MetadataExtractor()
        with patch("pathlib.Path.exists", return_value=True):
            # File not found (at the start of method)
            with patch("pathlib.Path.exists", return_value=False):
                with pytest.raises(FileNotFoundError):
                    extractor.extract_metadata_all("file.jpg")

            # CalledProcessError
            with patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "cmd", stderr="error"),
            ):
                assert extractor.extract_metadata_all("file.jpg") == {}

            # Timeout
            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)):
                assert extractor.extract_metadata_all("file.jpg") == {}

            # JSON error
            mock_result = MagicMock()
            mock_result.stdout = "invalid json"
            with patch("subprocess.run", return_value=mock_result):
                assert extractor.extract_metadata_all("file.jpg") == {}

            # Empty result
            mock_result.stdout = "[]"
            with patch("subprocess.run", return_value=mock_result):
                assert extractor.extract_metadata_all("file.jpg") == {}
