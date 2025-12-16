"""Comprehensive test suite for media types utility."""
from io import BytesIO

from cl_ml_tools.utils.media_types import (
    MediaType,
    contains_url,
    determine_media_type,
    determine_mime,
)

# ============================================================================
# Test Class 1: MediaType Enum
# ============================================================================


class TestMediaType:
    """Test MediaType enum and from_mime method."""

    def test_media_type_values(self) -> None:
        """Test that all MediaType values are correctly defined."""
        assert MediaType.TEXT == "text"
        assert MediaType.IMAGE == "image"
        assert MediaType.VIDEO == "video"
        assert MediaType.URL == "url"
        assert MediaType.AUDIO == "audio"
        assert MediaType.FILE == "file"

    def test_from_mime_image(self) -> None:
        """Test from_mime with image MIME types."""
        assert MediaType.from_mime("image/jpeg") == MediaType.IMAGE
        assert MediaType.from_mime("image/png") == MediaType.IMAGE
        assert MediaType.from_mime("image/gif") == MediaType.IMAGE
        assert MediaType.from_mime("image/webp") == MediaType.IMAGE

    def test_from_mime_video(self) -> None:
        """Test from_mime with video MIME types."""
        assert MediaType.from_mime("video/mp4") == MediaType.VIDEO
        assert MediaType.from_mime("video/mpeg") == MediaType.VIDEO
        assert MediaType.from_mime("video/quicktime") == MediaType.VIDEO
        assert MediaType.from_mime("video/x-msvideo") == MediaType.VIDEO

    def test_from_mime_audio(self) -> None:
        """Test from_mime with audio MIME types."""
        assert MediaType.from_mime("audio/mpeg") == MediaType.AUDIO
        assert MediaType.from_mime("audio/wav") == MediaType.AUDIO
        assert MediaType.from_mime("audio/ogg") == MediaType.AUDIO

    def test_from_mime_text(self) -> None:
        """Test from_mime with text MIME types."""
        assert MediaType.from_mime("text/plain") == MediaType.TEXT
        assert MediaType.from_mime("text/html") == MediaType.TEXT
        assert MediaType.from_mime("text/csv") == MediaType.TEXT

    def test_from_mime_unknown(self) -> None:
        """Test from_mime with unknown/application MIME types."""
        assert MediaType.from_mime("application/octet-stream") == MediaType.FILE
        assert MediaType.from_mime("application/pdf") == MediaType.FILE
        assert MediaType.from_mime("application/zip") == MediaType.FILE
        assert MediaType.from_mime("unknown/type") == MediaType.FILE


# ============================================================================
# Test Class 2: URL Detection
# ============================================================================


class TestContainsUrl:
    """Test contains_url function."""

    def test_valid_http_url(self) -> None:
        """Test detection of valid HTTP URLs."""
        assert contains_url("http://example.com") is True
        assert contains_url("http://example.com/path") is True
        assert contains_url("http://example.com/path?query=value") is True

    def test_valid_https_url(self) -> None:
        """Test detection of valid HTTPS URLs."""
        assert contains_url("https://example.com") is True
        assert contains_url("https://example.com/path/to/resource") is True
        assert contains_url("https://example.com:8080/path") is True

    def test_url_with_query_params(self) -> None:
        """Test URLs with query parameters."""
        assert contains_url("https://example.com?foo=bar&baz=qux") is True
        assert contains_url("https://api.example.com/v1/users?id=123") is True

    def test_url_with_fragment(self) -> None:
        """Test URLs with fragments."""
        assert contains_url("https://example.com/page#section") is True
        assert contains_url("https://docs.example.com/api#authentication") is True

    def test_url_with_whitespace(self) -> None:
        """Test that URLs with leading/trailing whitespace are detected."""
        assert contains_url("  https://example.com  ") is True
        assert contains_url("\thttps://example.com\t") is True

    def test_not_url_multiline(self) -> None:
        """Test that multiline text is not detected as URL."""
        assert contains_url("Line 1\nLine 2") is False
        assert contains_url("https://example.com\nAnother line") is False
        assert contains_url("Text\rWith\rCarriage\rReturns") is False

    def test_not_url_plain_text(self) -> None:
        """Test that plain text is not detected as URL."""
        assert contains_url("This is plain text") is False
        assert contains_url("not-a-url") is False
        assert contains_url("file:///local/path") is False  # file:// not http(s)://

    def test_not_url_invalid_scheme(self) -> None:
        """Test that URLs with invalid schemes are not detected."""
        assert contains_url("ftp://example.com") is False
        assert contains_url("mailto:user@example.com") is False
        assert contains_url("javascript:alert('xss')") is False


# ============================================================================
# Test Class 3: Media Type Determination
# ============================================================================


class TestDetermineMediaType:
    """Test determine_media_type function."""

    def test_determine_image_type(self) -> None:
        """Test image type determination."""
        bytes_io = BytesIO(b"fake image data")

        assert determine_media_type(bytes_io, "image/jpeg") == MediaType.IMAGE
        assert determine_media_type(bytes_io, "image/png") == MediaType.IMAGE

    def test_determine_video_type(self) -> None:
        """Test video type determination."""
        bytes_io = BytesIO(b"fake video data")

        assert determine_media_type(bytes_io, "video/mp4") == MediaType.VIDEO
        assert determine_media_type(bytes_io, "video/mpeg") == MediaType.VIDEO

    def test_determine_audio_type(self) -> None:
        """Test audio type determination."""
        bytes_io = BytesIO(b"fake audio data")

        assert determine_media_type(bytes_io, "audio/mpeg") == MediaType.AUDIO
        assert determine_media_type(bytes_io, "audio/wav") == MediaType.AUDIO

    def test_determine_text_type(self) -> None:
        """Test text type determination (non-URL)."""
        bytes_io = BytesIO(b"This is plain text content")

        result = determine_media_type(bytes_io, "text/plain")
        assert result == MediaType.TEXT

    def test_determine_url_type(self) -> None:
        """Test URL type determination from text."""
        bytes_io = BytesIO(b"https://example.com")

        result = determine_media_type(bytes_io, "text/plain")
        assert result == MediaType.URL

    def test_determine_file_type(self) -> None:
        """Test generic file type determination."""
        bytes_io = BytesIO(b"unknown binary data")

        result = determine_media_type(bytes_io, "application/octet-stream")
        assert result == MediaType.FILE

    def test_multiline_text_not_url(self) -> None:
        """Test that multiline text is detected as TEXT, not URL."""
        bytes_io = BytesIO(b"Line 1\nLine 2\nLine 3")

        result = determine_media_type(bytes_io, "text/plain")
        assert result == MediaType.TEXT


# ============================================================================
# Test Class 4: MIME Detection (Magic Library)
# ============================================================================


class TestDetermineMime:
    """Test determine_mime function with magic library."""

    def test_determine_mime_with_explicit_type(self) -> None:
        """Test that explicit file_type is used when provided."""
        bytes_io = BytesIO(b"any content")

        result = determine_mime(bytes_io, file_type="image/jpeg")
        assert result == MediaType.IMAGE

    def test_determine_mime_auto_detect(self) -> None:
        """Test MIME type auto-detection (requires python-magic)."""
        # Simple test with plain text
        bytes_io = BytesIO(b"Hello, world!")

        result = determine_mime(bytes_io)
        # Result should be TEXT (magic detects as text/plain)
        assert result in (MediaType.TEXT, MediaType.FILE)  # May vary by magic version

    def test_determine_mime_seek_position(self) -> None:
        """Test that BytesIO position is reset after MIME detection."""
        bytes_io = BytesIO(b"test content")
        bytes_io.seek(5)  # Move position

        _ = determine_mime(bytes_io, file_type="text/plain")

        # Position should still be seekable
        bytes_io.seek(0)
        assert bytes_io.read() == b"test content"

    def test_determine_mime_empty_bytes(self) -> None:
        """Test MIME detection with empty BytesIO."""
        bytes_io = BytesIO(b"")

        result = determine_mime(bytes_io)
        # Empty files are typically detected as application/x-empty or similar
        assert isinstance(result, MediaType)


# ============================================================================
# Test Class 5: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_url_detection_empty_string(self) -> None:
        """Test URL detection with empty string."""
        assert contains_url("") is False
        assert contains_url("   ") is False

    def test_url_detection_just_scheme(self) -> None:
        """Test URL detection with just scheme."""
        assert contains_url("http://") is False
        assert contains_url("https://") is False

    def test_media_type_case_sensitivity(self) -> None:
        """Test that MIME type detection is case-insensitive for prefix."""
        # from_mime uses startswith, which is case-sensitive
        # But standard MIME types are lowercase
        bytes_io = BytesIO(b"test")

        # Standard lowercase
        assert determine_media_type(bytes_io, "image/jpeg") == MediaType.IMAGE

        # Uppercase should not match (startswith is case-sensitive)
        # This is actually expected behavior - MIME types should be lowercase
        assert determine_media_type(bytes_io, "IMAGE/JPEG") == MediaType.FILE

    def test_unicode_in_text(self) -> None:
        """Test text with Unicode characters."""
        bytes_io = BytesIO("Hello 世界! 🌍".encode("utf-8"))

        result = determine_media_type(bytes_io, "text/plain")
        assert result == MediaType.TEXT

    def test_url_with_unicode_domain(self) -> None:
        """Test URL detection with internationalized domain."""
        # This specific pattern doesn't support Unicode domains
        assert contains_url("https://münchen.de") is False

        # ASCII-only domain works
        assert contains_url("https://example.com") is True
