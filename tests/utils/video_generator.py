"""Generate synthetic test videos for HLS streaming tests.

Uses the existing RandomMediaGenerator to create videos with sufficient
duration and quality to generate multiple HLS .ts segments.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_hls_test_video(
    output_path: Path,
    duration_seconds: int = 30,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> Path:
    """Generate a high-quality test video suitable for HLS streaming tests.

    Creates a video with multiple colored scenes that will generate multiple
    .ts segments when processed by FFmpeg for HLS streaming.

    Args:
        output_path: Path where video should be saved
        duration_seconds: Video duration in seconds (default: 30 for multiple segments)
        width: Video width in pixels (default: 1280 for HD quality)
        height: Video height in pixels (default: 720 for HD quality)
        fps: Frames per second (default: 30)

    Returns:
        Path to generated video

    Raises:
        RuntimeError: If video generation fails
        ImportError: If cv2 (OpenCV) is not available
    """
    try:
        from cl_ml_tools.utils.random_media_generator.scene_generator import SceneGenerator
        from cl_ml_tools.utils.random_media_generator.video_generator import VideoGenerator
    except ImportError as e:
        raise ImportError(
            "OpenCV (cv2) is required for video generation. "
            + "Install with: pip install opencv-python"
        ) from e

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create multiple scenes for visual variety
    # Each scene is ~5 seconds, so 30 seconds = 6 scenes
    scenes_per_video = duration_seconds // 5
    from cl_ml_tools.utils.random_media_generator.scene_generator import SceneGenerator

    scenes: list[SceneGenerator] = []

    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]

    for i in range(scenes_per_video):
        color = colors[i % len(colors)]
        scene = SceneGenerator(
            duration_seconds=5,
            background_color=color,
            num_shapes=0,
        )
        scenes.append(scene)

    try:
        logger.info(
            f"Generating test video: {output_path} "
            + f"({duration_seconds}s, {width}x{height} @ {fps}fps)"
        )

        generator = VideoGenerator(
            out_dir=str(output_path.parent),
            fileName=output_path.stem,
            MIMEType="video/mp4",
            width=width,
            height=height,
            fps=fps,
            scenes=scenes,
        )

        generator.generate()

        if not output_path.exists():
            raise RuntimeError(f"Video generation failed: {output_path} was not created")

        file_size = output_path.stat().st_size
        logger.info(
            f"Generated test video: {output_path} " + f"({file_size:,} bytes, {duration_seconds}s)"
        )

        return output_path

    except Exception as exc:
        logger.error(f"Failed to generate video: {exc}")
        raise RuntimeError(f"Video generation failed: {exc}") from exc


def ensure_hls_test_videos_exist(videos_dir: Path, count: int = 2) -> list[Path]:
    """Ensure HLS test videos exist, generating them if needed.

    Args:
        videos_dir: Directory where videos should be stored
        count: Number of test videos to ensure exist

    Returns:
        List of paths to available test videos
    """
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing videos
    existing = list(videos_dir.glob("*.mp4"))
    if len(existing) >= count:
        logger.info(f"Found {len(existing)} existing test videos")
        return existing[:count]

    # Generate missing videos
    needed = count - len(existing)
    logger.info(f"Generating {needed} test videos for HLS streaming tests")
    videos = existing.copy()
    for i in range(len(existing), count):
        video_path = videos_dir / f"test_video_{i + 1}.mp4"
        try:
            # Create longer videos with higher quality for HLS segmentation
            # 30 seconds at 1280x720 should generate multiple .ts segments
            generated = generate_hls_test_video(
                output_path=video_path,
                duration_seconds=30,
                width=1280,
                height=720,
                fps=30,
            )
            videos.append(generated)
        except (RuntimeError, ImportError) as exc:
            logger.warning(f"Failed to generate video {i + 1}: {exc}")
            # If we can't generate, just return what we have
            break

    return videos
