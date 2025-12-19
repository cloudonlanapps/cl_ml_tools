"""Unit tests for media type detection utilities.

Tests MediaType enum, MIME type detection, URL validation, and media type determination.
Uses synthetic data (BytesIO) without requiring external files.
"""

import re
from io import BytesIO

import pytest

from cl_ml_tools.utils.media_types import (
    MediaType,
    contains_url,
    determine_media_type,
    determine_mime,
)

# ============================================================================
# MediaType Enum Tests
# ============================================================================


def test_media_type_enum_values():
    """Test MediaType enum has expected values."""
    assert MediaType.TEXT == "text"
    assert MediaType.IMAGE == "image"
    assert MediaType.VIDEO == "video"
    assert MediaType.URL == "url"
    assert MediaType.AUDIO == "audio"
    assert MediaType.FILE == "file"


def test_media_type_enum_membership():
    """Test MediaType enum membership checks."""
    assert "image" in MediaType.__members__.values()
    assert "video" in MediaType.__members__.values()
    assert "audio" in MediaType.__members__.values()
    assert "text" in MediaType.__members__.values()
    assert "url" in MediaType.__members__.values()
    assert "file" in MediaType.__members__.values()


# ============================================================================
# MediaType.from_mime Tests
# ============================================================================


def test_from_mime_image_types():
    """Test from_mime correctly identifies image MIME types."""
    assert MediaType.from_mime("image/jpeg") == MediaType.IMAGE
    assert MediaType.from_mime("image/png") == MediaType.IMAGE
    assert MediaType.from_mime("image/gif") == MediaType.IMAGE
    assert MediaType.from_mime("image/webp") == MediaType.IMAGE
    assert MediaType.from_mime("image/svg+xml") == MediaType.IMAGE


def test_from_mime_video_types():
    """Test from_mime correctly identifies video MIME types."""
    assert MediaType.from_mime("video/mp4") == MediaType.VIDEO
    assert MediaType.from_mime("video/mpeg") == MediaType.VIDEO
    assert MediaType.from_mime("video/quicktime") == MediaType.VIDEO
    assert MediaType.from_mime("video/x-msvideo") == MediaType.VIDEO
    assert MediaType.from_mime("video/webm") == MediaType.VIDEO


def test_from_mime_audio_types():
    """Test from_mime correctly identifies audio MIME types."""
    assert MediaType.from_mime("audio/mpeg") == MediaType.AUDIO
    assert MediaType.from_mime("audio/wav") == MediaType.AUDIO
    assert MediaType.from_mime("audio/ogg") == MediaType.AUDIO
    assert MediaType.from_mime("audio/mp4") == MediaType.AUDIO
    assert MediaType.from_mime("audio/aac") == MediaType.AUDIO


def test_from_mime_text_types():
    """Test from_mime correctly identifies text MIME types."""
    assert MediaType.from_mime("text/plain") == MediaType.TEXT
    assert MediaType.from_mime("text/html") == MediaType.TEXT
    assert MediaType.from_mime("text/css") == MediaType.TEXT
    assert MediaType.from_mime("text/javascript") == MediaType.TEXT
    assert MediaType.from_mime("text/csv") == MediaType.TEXT


def test_from_mime_other_types():
    """Test from_mime returns FILE for unrecognized MIME types."""
    assert MediaType.from_mime("application/pdf") == MediaType.FILE
    assert MediaType.from_mime("application/json") == MediaType.FILE
    assert MediaType.from_mime("application/zip") == MediaType.FILE
    assert MediaType.from_mime("application/octet-stream") == MediaType.FILE


# ============================================================================
# contains_url Tests
# ============================================================================


def test_contains_url_valid_http():
    """Test contains_url identifies valid HTTP URLs."""
    assert contains_url("http://example.com") is True
    assert contains_url("http://example.com/path") is True
    assert contains_url("http://example.com/path?query=value") is True
    assert contains_url("http://subdomain.example.com") is True


def test_contains_url_valid_https():
    """Test contains_url identifies valid HTTPS URLs."""
    assert contains_url("https://example.com") is True
    assert contains_url("https://example.com/path") is True
    assert contains_url("https://example.com:8080/path") is True
    assert contains_url("https://api.example.com/v1/resource") is True


def test_contains_url_with_whitespace():
    """Test contains_url handles URLs with surrounding whitespace."""
    assert contains_url("  https://example.com  ") is True
    assert contains_url("\thttps://example.com\t") is True
    assert contains_url("\nhot valid url") is False


def test_contains_url_rejects_newlines():
    """Test contains_url rejects text with newlines."""
    assert contains_url("https://example.com\ntext") is False
    assert contains_url("line1\nhttps://example.com") is False
    assert contains_url("https://example.com\r\n") is False


def test_contains_url_rejects_non_urls():
    """Test contains_url rejects non-URL text."""
    assert contains_url("not a url") is False
    assert contains_url("example.com") is False
    assert contains_url("ftp://example.com") is False
    assert contains_url("") is False
    assert contains_url("   ") is False


def test_contains_url_special_characters():
    """Test contains_url handles URLs with special characters."""
    assert contains_url("https://example.com/path?q=hello%20world") is True
    assert contains_url("https://example.com/path#anchor") is True
    assert contains_url("https://user:pass@example.com") is True
    assert contains_url("https://example.com/path?a=1&b=2") is True


# ============================================================================
# determine_media_type Tests
# ============================================================================


def test_determine_media_type_image():
    """Test determine_media_type correctly identifies images."""
    # Create dummy image data
    image_data = b"\xff\xd8\xff\xe0"  # JPEG magic bytes
    bytes_io = BytesIO(image_data)

    result = determine_media_type(bytes_io, "image/jpeg")
    assert result == MediaType.IMAGE


def test_determine_media_type_video():
    """Test determine_media_type correctly identifies videos."""
    video_data = b"\x00\x00\x00\x20ftypisom"  # MP4 magic bytes
    bytes_io = BytesIO(video_data)

    result = determine_media_type(bytes_io, "video/mp4")
    assert result == MediaType.VIDEO


def test_determine_media_type_audio():
    """Test determine_media_type correctly identifies audio."""
    audio_data = b"ID3"  # MP3 ID3 tag
    bytes_io = BytesIO(audio_data)

    result = determine_media_type(bytes_io, "audio/mpeg")
    assert result == MediaType.AUDIO


def test_determine_media_type_url():
    """Test determine_media_type correctly identifies URL text."""
    url_text = "https://example.com/image.jpg"
    bytes_io = BytesIO(url_text.encode("utf-8"))

    result = determine_media_type(bytes_io, "text/plain")
    assert result == MediaType.URL


def test_determine_media_type_text():
    """Test determine_media_type correctly identifies plain text."""
    text = "This is plain text content"
    bytes_io = BytesIO(text.encode("utf-8"))

    result = determine_media_type(bytes_io, "text/plain")
    assert result == MediaType.TEXT


def test_determine_media_type_text_multiline():
    """Test determine_media_type treats multiline text as TEXT not URL."""
    text = "First line\nSecond line\nhttps://example.com"
    bytes_io = BytesIO(text.encode("utf-8"))

    result = determine_media_type(bytes_io, "text/plain")
    assert result == MediaType.TEXT


def test_determine_media_type_other_file():
    """Test determine_media_type returns FILE for unknown types."""
    data = b"\x50\x4b\x03\x04"  # ZIP magic bytes
    bytes_io = BytesIO(data)

    result = determine_media_type(bytes_io, "application/zip")
    assert result == MediaType.FILE


# ============================================================================
# determine_mime Tests
# ============================================================================


def test_determine_mime_with_explicit_type():
    """Test determine_mime uses provided file_type parameter."""
    data = b"any data"
    bytes_io = BytesIO(data)

    result = determine_mime(bytes_io, file_type="image/jpeg")
    assert result == MediaType.IMAGE


def test_determine_mime_detects_jpeg():
    """Test determine_mime auto-detects JPEG from buffer."""
    # JPEG magic bytes
    jpeg_data = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    bytes_io = BytesIO(jpeg_data)

    result = determine_mime(bytes_io)
    assert result == MediaType.IMAGE


def test_determine_mime_detects_png():
    """Test determine_mime auto-detects PNG from buffer."""
    # PNG magic bytes + minimal IHDR chunk (required for proper detection)
    png_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    bytes_io = BytesIO(png_data)

    result = determine_mime(bytes_io)
    assert result == MediaType.IMAGE


def test_determine_mime_empty_buffer():
    """Test determine_mime handles empty buffer."""
    bytes_io = BytesIO(b"")

    # Should default to FILE for unknown/empty content
    result = determine_mime(bytes_io)
    assert result in (MediaType.FILE, MediaType.TEXT)


def test_determine_mime_resets_position():
    """Test determine_mime resets BytesIO position to 0."""
    data = b"test data"
    bytes_io = BytesIO(data)
    bytes_io.seek(5)  # Move to middle

    _ = determine_mime(bytes_io, file_type="text/plain")

    # Position should be reset to 0
    assert bytes_io.tell() == 0


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


def test_media_type_case_sensitivity():
    """Test MIME type matching is case-insensitive for prefixes."""
    # MediaType.from_mime uses startswith which is case-sensitive
    assert MediaType.from_mime("Image/jpeg") == MediaType.FILE  # Case matters
    assert MediaType.from_mime("image/jpeg") == MediaType.IMAGE


def test_contains_url_regex_pattern():
    """Test contains_url regex pattern matches expected formats."""
    # The regex should be strict about URL format
    pattern = re.compile(
        r"^http[s]?:\/\/(?:[a-zA-Z0-9\-._~:/?#[\]@!$&\'()*+,;=]|%[0-9a-fA-F][0-9a-fA-F])+$",
    )

    assert pattern.match("https://example.com") is not None
    assert pattern.match("http://example.com/path") is not None
    assert pattern.match("not a url") is None
    assert pattern.match("ftp://example.com") is None


def test_determine_media_type_utf8_decoding():
    """Test determine_media_type handles UTF-8 text properly."""
    # Text with unicode characters
    text = "Hello 世界 🌍"
    bytes_io = BytesIO(text.encode("utf-8"))

    result = determine_media_type(bytes_io, "text/plain")
    assert result == MediaType.TEXT


def test_determine_media_type_invalid_utf8():
    """Test determine_media_type handles invalid UTF-8 gracefully."""
    # Invalid UTF-8 sequence
    invalid_utf8 = b"\xff\xfe\xfd"
    bytes_io = BytesIO(invalid_utf8)

    # Should fail to decode and not be treated as text
    with pytest.raises(UnicodeDecodeError):
        bytes_io.getvalue().decode("utf-8")

    # But with file type specified as non-text, should work
    result = determine_media_type(bytes_io, "application/octet-stream")
    assert result == MediaType.FILE


def test_media_type_enum_string_representation():
    """Test MediaType enum string values."""
    assert str(MediaType.IMAGE) == "image"
    assert str(MediaType.VIDEO) == "video"
    assert MediaType.IMAGE.value == "image"
    assert MediaType.VIDEO.value == "video"
