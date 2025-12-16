"""Media resize task implementation."""

from io import BytesIO
from pathlib import Path
from typing import Callable, override

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from ...utils.media_types import MediaType, determine_mime
from .algo.image_resize import image_resize
from .algo.video_resize import video_resize
from .schema import MediaResizeParams


class MediaResizeTask(ComputeModule[MediaResizeParams]):
    """Compute module for resizing images and videos."""

    @property
    @override
    def task_type(self) -> str:
        return "media_resize"

    @override
    def get_schema(self) -> type[BaseJobParams]:
        return MediaResizeParams

    @override
    async def execute(
        self,
        job: Job,
        params: MediaResizeParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        try:
            processed_files: list[str] = []
            media_types: list[str] = []
            total_files = len(params.input_paths)

            for index, (input_path, output_path) in enumerate(
                zip(params.input_paths, params.output_paths)
            ):
                # Detect media type
                input_file = Path(input_path)
                if not input_file.exists():
                    raise FileNotFoundError(f"Input file not found: {input_path}")

                with open(input_file, "rb") as f:
                    bytes_io = BytesIO(f.read())
                    media_type = determine_mime(bytes_io)

                # Route to appropriate resize function
                if media_type == MediaType.IMAGE:
                    output = image_resize(
                        input_path=input_path,
                        output_path=output_path,
                        width=params.width,
                        height=params.height,
                        maintain_aspect_ratio=params.maintain_aspect_ratio,
                    )
                    media_types.append("image")

                elif media_type == MediaType.VIDEO:
                    output = video_resize(
                        input_path=input_path,
                        output_path=output_path,
                        width=params.width,
                        height=params.height,
                    )
                    media_types.append("video")

                else:
                    return {
                        "status": "error",
                        "error": f"Unsupported media type: {media_type}. Only image and video are supported.",
                    }

                processed_files.append(output)

                if progress_callback:
                    progress = int((index + 1) / total_files * 100)
                    progress_callback(progress)

            return {
                "status": "ok",
                "task_output": {
                    "processed_files": processed_files,
                    "media_types": media_types,
                    "dimensions": {
                        "width": params.width,
                        "height": params.height,
                    },
                },
            }

        except ImportError as e:
            return {
                "status": "error",
                "error": f"Required library not installed: {e}",
            }
        except FileNotFoundError as e:
            return {"status": "error", "error": str(e)}
        except Exception as e:
            return {"status": "error", "error": str(e)}
