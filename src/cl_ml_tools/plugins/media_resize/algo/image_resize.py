"""Pure image resize computation logic (single file)."""

from pathlib import Path

from PIL import Image


def image_resize(
    *,
    input_path: str | Path,
    output_path: str | Path,
    width: int | None = None,
    height: int | None = None,
    maintain_aspect_ratio: bool = True,
) -> str:
    """
    Resize a single image and write output.

    Framework-agnostic, single-responsibility function.

    Args:
        input_path: Path to input image
        output_path: Path to output image
        width: Target width
        height: Target height
        maintain_aspect_ratio: Preserve aspect ratio if True

    Returns:
        Output file path as string

    Raises:
        FileNotFoundError: If input image does not exist
        OSError: If Pillow fails to read/write the image
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Default to 256x256 if neither specified
    if width is None and height is None:
        width = 256
        height = 256
    elif width is None:
        # Only height specified, use it for both (PIL thumbnail maintains aspect)
        width = height
    elif height is None:
        # Only width specified, use it for both (PIL thumbnail maintains aspect)
        height = width
    # If both specified, use as-is (PIL thumbnail will fit within box)

    with Image.open(input_path) as img:
        if maintain_aspect_ratio:
            img.thumbnail((width, height), Image.Resampling.LANCZOS)
            resized = img
        else:
            resized = img.resize(
                (width, height),
                Image.Resampling.LANCZOS,
            )

        resized.save(output_path)

    return str(output_path)
