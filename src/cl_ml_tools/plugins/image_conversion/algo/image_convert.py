"""Pure image conversion computation logic (single file)."""

from pathlib import Path

from PIL import Image


def image_convert(
    *,
    input_path: str | Path,
    output_path: str | Path,
    format: str,
    quality: int | None = None,
) -> str:
    """
    Convert a single image to a different format.

    Framework-agnostic, single-image operation.

    Args:
        input_path: Path to input image
        output_path: Path to output image
        format: Target image format (jpg, png, webp, etc.)
        quality: Optional quality parameter (for supported formats)

    Returns:
        Output file path as string

    Raises:
        FileNotFoundError: If input file does not exist
        OSError: If Pillow fails to read/write image
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    with Image.open(input_path) as img:
        fmt = format.lower()

        # JPEG does not support alpha channel
        if fmt in ("jpg", "jpeg") and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        save_kwargs: dict[str, object] = {}

        if fmt in ("jpg", "jpeg", "webp") and quality is not None:
            save_kwargs["quality"] = quality

        if fmt == "png":
            save_kwargs["optimize"] = True

        # Ensure parent directory exists (caller's responsibility to create)
        if not output_path.parent.exists():
            raise FileNotFoundError(f"Output directory does not exist: {output_path.parent}")

        img.save(
            output_path,
            format=get_pil_format(fmt),
            **save_kwargs,
        )

    return str(output_path)


def get_pil_format(format_str: str) -> str:
    """Convert format string to PIL format name."""
    format_map = {
        "jpg": "JPEG",
        "jpeg": "JPEG",
        "png": "PNG",
        "webp": "WEBP",
        "gif": "GIF",
        "bmp": "BMP",
        "tiff": "TIFF",
    }
    return format_map.get(format_str.lower(), format_str.upper())
