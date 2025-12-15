"""Image resize task implementation."""

from typing import Callable, override

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from .algo.image_resize import image_resize
from .schema import ImageResizeParams


class ImageResizeTask(ComputeModule[ImageResizeParams]):
    """Compute module for resizing images."""

    @property
    @override
    def task_type(self) -> str:
        return "image_resize"

    @override
    def get_schema(self) -> type[BaseJobParams]:
        return ImageResizeParams

    @override
    async def execute(
        self,
        job: Job,
        params: ImageResizeParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        try:
            processed_files: list[str] = []
            total_files = len(params.input_paths)

            for index, (input_path, output_path) in enumerate(
                zip(params.input_paths, params.output_paths)
            ):
                output = image_resize(
                    input_path=input_path,
                    output_path=output_path,
                    width=params.width,
                    height=params.height,
                    maintain_aspect_ratio=params.maintain_aspect_ratio,
                )

                processed_files.append(output)

                if progress_callback:
                    progress = int((index + 1) / total_files * 100)
                    progress_callback(progress)

            return {
                "status": "ok",
                "task_output": {
                    "processed_files": processed_files,
                    "dimensions": {
                        "width": params.width,
                        "height": params.height,
                    },
                },
            }

        except ImportError:
            return {
                "status": "error",
                "error": "Pillow is not installed. Install with: pip install cl_ml_tools[compute]",
            }
        except FileNotFoundError as e:
            return {"status": "error", "error": f"Input file not found: {e}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
