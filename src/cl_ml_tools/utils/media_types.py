import re
from enum import StrEnum
from io import BytesIO

import mimetypes
import magic
from loguru import logger


class MediaType(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    URL = "url"
    AUDIO = "audio"
    FILE = "file"

    @classmethod
    def from_mime(cls, file_type: str) -> "MediaType":
        if file_type.startswith("image"):
            return MediaType.IMAGE
        elif file_type.startswith("video"):
            return MediaType.VIDEO
        elif file_type.startswith("audio"):
            return MediaType.AUDIO
        elif file_type.startswith("text"):
            """text = bytes_io.getvalue().decode("utf-8")
            if contains_url(text):
                return MediaType.URL
            else:"""
            return MediaType.TEXT
        else:
            return MediaType.FILE


def contains_url(text: str):
    if "\n" in text or "\r" in text:
        return False

    stripped_text = text.strip()

    url_pattern = re.compile(
        r"^http[s]?:\/\/(?:[a-zA-Z0-9\-._~:/?#[\]@!$&\'()*+,;=]|%[0-9a-fA-F][0-9a-fA-F])+$"
    )
    return bool(url_pattern.match(stripped_text))


def get_extension_from_mime(mime_type: str, media_type: MediaType) -> str:
    """Determine file extension from MIME type.

    Args:
        mime_type: MIME type string (e.g., "image/jpeg")
        media_type: MediaType enum value

    Returns:
        File extension without dot (e.g., "jpg")
    """
    # Common MIME type to extension mappings
    mime_to_ext: dict[str, str] = {
        # Images
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
        "image/bmp": "bmp",
        "image/tiff": "tiff",
        "image/svg+xml": "svg",
        "image/heic": "heic",
        "image/heif": "heif",
        # Videos
        "video/mp4": "mp4",
        "video/mpeg": "mpeg",
        "video/quicktime": "mov",
        "video/x-msvideo": "avi",
        "video/x-matroska": "mkv",
        "video/webm": "webm",
        "video/3gpp": "3gp",
        # Audio
        "audio/mpeg": "mp3",
        "audio/wav": "wav",
        "audio/ogg": "ogg",
        "audio/flac": "flac",
        "audio/aac": "aac",
        # Documents
        "application/pdf": "pdf",
        "text/plain": "txt",
    }

    if mime_type in mime_to_ext:
        return mime_to_ext[mime_type]

    # Try to extract from mime_type pattern (e.g., "image/jpeg" → "jpeg")
    if "/" in mime_type:
        subtype = mime_type.split("/")[1]
        # Remove any parameters (e.g., "jpeg; charset=utf-8" → "jpeg")
        subtype = subtype.split(";")[0].strip()
        if subtype:
            return subtype

    # Fallback based on media type
    match media_type:
        case MediaType.IMAGE:
            return "jpg"
        case MediaType.VIDEO:
            return "mp4"
        case MediaType.AUDIO:
            return "mp3"
        case MediaType.TEXT:
            return "txt"
        case _:
            return "bin"


def determine_mime(bytes_io: BytesIO, file_type: str | None = None) -> tuple[str, MediaType]:
    # Save original position

    try:
        if not file_type:
            _ = bytes_io.seek(0)
            buffer = bytes_io.read(2048)  # Read small chunk for detection
            
            # 1. Try python-magic (best but depends on libmagic)
            try:
                mime = magic.Magic(mime=True)
                file_type = mime.from_buffer(buffer)
            except Exception as e:
                logger.warning(f"python-magic failed: {e}. Falling back to mimetypes.")
                file_type = None

            # 2. Try mimetypes (no buffer content guessing usually, but we check if magic failed)
            if not file_type:
                # mimetypes doesn't guess from buffer content alone well, 
                # but it's a safe library import.
                file_type = "application/octet-stream"

        return file_type, MediaType.from_mime(file_type)
    finally:
        # Always reset position to 0 (as expected by tests)
        _ = bytes_io.seek(0)


def determine_media_type(bytes_io: BytesIO, file_type: str) -> MediaType:
    if file_type.startswith("image"):
        return MediaType.IMAGE
    elif file_type.startswith("video"):
        return MediaType.VIDEO
    elif file_type.startswith("audio"):
        return MediaType.AUDIO
    elif file_type.startswith("text"):
        text = bytes_io.getvalue().decode("utf-8")
        if contains_url(text):
            return MediaType.URL
        else:
            return MediaType.TEXT
    else:
        return MediaType.FILE
