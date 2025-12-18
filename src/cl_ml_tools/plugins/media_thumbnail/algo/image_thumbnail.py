"""Pure image thumbnail computation logic (single file)."""

from pathlib import Path

from PIL import Image


def image_thumbnail(
    *,
    input_path: str | Path,
    output_path: str | Path,
    width: int | None = None,
    height: int | None = None,
    maintain_aspect_ratio: bool = True,
) -> str:
    """
    Thumbnail a single image and write output.

    Framework-agnostic, single-responsibility function.

    Args:
        input_path: Path to input image
        output_path: Path to output image
        w: Target w
        h: Target h
        maintain_aspect_ratio: Preserve aspect ratio if True

    Returns:
        Output file path as string

    Raises:
        FileNotFoundError: If input image does not exist
        OSError: If Pillow fails to read/write the image
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    with Image.open(input_path) as img:
        original_width, original_height = img.size

        # Defaults
        DEFAULT_SIZE = 256

        # Normalize inputs
        w_in: int | None = width
        h_in: int | None = height
        h: int = h_in if h_in is not None else DEFAULT_SIZE
        w: int = w_in if w_in is not None else DEFAULT_SIZE

        if w_in is None and h_in is None:
            w = h = DEFAULT_SIZE

        elif w_in is None:
            h = h_in if h_in is not None else DEFAULT_SIZE
            if maintain_aspect_ratio:
                w = int(h * (original_width / original_height))
            else:
                w = h

        elif h_in is None:
            w = w_in
            if maintain_aspect_ratio:
                h = int(w * (original_height / original_width))
            else:
                h = w

        else:
            w = w_in
            h = h_in

        # Resize the image
        thumbnail = img.resize((w, h), Image.Resampling.LANCZOS)
        thumbnail.save(output_path)

    return str(output_path)
