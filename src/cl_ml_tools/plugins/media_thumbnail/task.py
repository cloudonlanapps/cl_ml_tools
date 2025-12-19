"""Media thumbnail task implementation."""

from io import BytesIO
from pathlib import Path
from typing import Callable, override

from ...common.compute_module import ComputeModule
from ...common.job_storage import JobStorage
from ...utils.media_types import MediaType, determine_mime
from .algo.image_thumbnail import image_thumbnail
from .algo.video_thumbnail import video_thumbnail
from .schema import MediaThumbnailOutput, MediaThumbnailParams


class MediaThumbnailTask(ComputeModule[MediaThumbnailParams, MediaThumbnailOutput]):
    """Compute module for generating a thumbnail for an image or video."""

    schema: type[MediaThumbnailParams] = MediaThumbnailParams

    @property
    @override
    def task_type(self) -> str:
        return "media_thumbnail"

    @override
    async def run(
        self,
        job_id: str,
        params: MediaThumbnailParams,
        storage: JobStorage,
        progress_callback: Callable[[int], None] | None = None,
    ) -> MediaThumbnailOutput:
        input_path = storage.resolve_path(job_id, params.input_path)
        if not input_path.exists():
            raise FileNotFoundError("Input file not found: " + str(input_path))

        try:
            with input_path.open("rb") as f:
                media_type = determine_mime(BytesIO(f.read()))
        except ImportError as exc:
            raise RuntimeError("Required library not installed: " + str(exc)) from exc

        # Validate dimensions
        if params.width is not None and params.width <= 0:
            raise ValueError(f"Width must be positive, got {params.width}")
        if params.height is not None and params.height <= 0:
            raise ValueError(f"Height must be positive, got {params.height}")

        output_path = Path(
            storage.allocate_path(
                job_id=job_id,
                relative_path=params.output_path,
            )
        )

        if media_type == MediaType.IMAGE:
            _ = image_thumbnail(
                input_path=str(input_path),
                output_path=output_path,
                width=params.width,
                height=params.height,
                maintain_aspect_ratio=params.maintain_aspect_ratio,
            )
            media_type_str = "image"

        elif media_type == MediaType.VIDEO:
            _ = video_thumbnail(
                input_path=params.input_path,
                output_path=output_path,
                width=params.width,
                height=params.height,
            )
            media_type_str = "video"

        else:
            raise RuntimeError(
                "Unsupported media type: "
                + str(media_type)
                + ". Only image and video are supported."
            )

        if progress_callback:
            progress_callback(100)

        return MediaThumbnailOutput(
            media_type=media_type_str,
        )
