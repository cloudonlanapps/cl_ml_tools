"""Video thumbnail computation logic using FFmpeg."""

import subprocess
from pathlib import Path


def video_thumbnail(
    *,
    input_path: str | Path,
    output_path: str | Path,
    width: int | None = None,
    height: int | None = None,
) -> str:
    """
    Thumbnail a video using FFmpeg with aspect ratio maintenance.

    Creates a 4x4 tiled thumbnail grid from keyframes.

    Args:
        input_path: Path to input video
        output_path: Path to output image (JPG)
        width: Target width (None = use default 256)
        height: Target height (None = use default 256)

    Returns:
        Output file path as string

    Raises:
        FileNotFoundError: If input video does not exist
        RuntimeError: If FFmpeg command fails
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not output_path.parent.exists():
        raise FileNotFoundError(f"Output directory not found: {output_path.parent}")

    # Default to 256x256 if neither specified
    if width is None and height is None:
        width = 256
        height = 256

    # Build scale filter based on provided dimensions
    if width is not None and height is not None:
        # Both specified: fit within box, maintain aspect ratio
        scale_filter = (
            f"scale='min(iw,{width})':'min(ih,{height})':force_original_aspect_ratio=decrease"
        )
    elif width is not None:
        # Only width specified: maintain aspect ratio
        scale_filter = f"scale={width}:-1"
    else:  # height is not None
        # Only height specified: maintain aspect ratio
        scale_filter = f"scale=-1:{height}"

    tile_size = (4, 4)

    ffmpeg_command = [
        "ffmpeg",
        "-loglevel",
        "panic",
        "-y",
        "-skip_frame",
        "nokey",
        "-i",
        str(input_path),
        "-frames",
        "1",
        "-q:v",
        "1",
        "-vf",
        f"tile={tile_size[0]}x{tile_size[1]},loop={tile_size[0] * tile_size[1]}:1,{scale_filter}",
        str(output_path),
    ]

    result = subprocess.run(
        ffmpeg_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg command failed: {result.stderr}")

    return str(output_path)
