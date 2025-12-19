"""Image conversion task implementation."""

from typing import Callable, override

from ...common.compute_module import ComputeModule
from ...common.job_storage import JobStorage
from .algo.image_convert import image_convert
from .schema import ImageConversionOutput, ImageConversionParams


class ImageConversionTask(ComputeModule[ImageConversionParams, ImageConversionOutput]):
    """Compute module for converting an image between formats."""

    schema: type[ImageConversionParams] = ImageConversionParams

    @property
    @override
    def task_type(self) -> str:
        return "image_conversion"

    @override
    async def run(
        self,
        job_id: str,
        params: ImageConversionParams,
        storage: JobStorage,
        progress_callback: Callable[[int], None] | None = None,
    ) -> ImageConversionOutput:
        input_path = storage.resolve_path(job_id, params.input_path)
        output_path = storage.allocate_path(
            job_id=job_id,
            relative_path=params.output_path,
        )

        try:
            _ = image_convert(
                input_path=str(input_path),
                output_path=output_path,
                format=params.format,
                quality=params.quality,
            )

        except ImportError as exc:
            raise RuntimeError(
                "Pillow is not installed. "
                + "Install with: pip install cl_ml_tools[compute]"
            ) from exc

        except FileNotFoundError as exc:
            raise FileNotFoundError("Input file not found: " + str(exc)) from exc

        if progress_callback:
            progress_callback(100)

        return ImageConversionOutput()
