"""Image conversion task implementation."""

from typing import Callable, override

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from .algo.image_convert import image_convert
from .schema import ImageConversionParams


class ImageConversionTask(ComputeModule[ImageConversionParams]):
    """Compute module for converting images between formats."""

    @property
    @override
    def task_type(self) -> str:
        return "image_conversion"

    @override
    def get_schema(self) -> type[BaseJobParams]:
        return ImageConversionParams

    @override
    async def execute(
        self,
        job: Job,
        params: ImageConversionParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        try:
            processed_files: list[str] = []
            total_files = len(params.input_paths)

            for index, (input_path, output_path) in enumerate(
                zip(params.input_paths, params.output_paths)
            ):
                output = image_convert(
                    input_path=input_path,
                    output_path=output_path,
                    format=params.format,
                    quality=params.quality,
                )

                processed_files.append(output)

                if progress_callback:
                    progress = int((index + 1) / total_files * 100)
                    progress_callback(progress)

            return {
                "status": "ok",
                "task_output": {
                    "processed_files": processed_files,
                    "format": params.format,
                    "quality": params.quality,
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
