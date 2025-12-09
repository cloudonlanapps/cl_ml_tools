"""Image resize task implementation."""

from typing import Callable, Optional, Dict, Any, Type

from cl_ml_tools.common.compute_module import ComputeModule
from cl_ml_tools.common.schemas import Job, BaseJobParams
from .schema import ImageResizeParams


class ImageResizeTask(ComputeModule):
    """Compute module for resizing images.

    Resizes images to specified dimensions using Pillow.
    """

    @property
    def task_type(self) -> str:
        """Return task type identifier."""
        return "image_resize"

    def get_schema(self) -> Type[BaseJobParams]:
        """Return the Pydantic params class for this task."""
        return ImageResizeParams

    async def execute(
        self,
        job: Job,
        params: ImageResizeParams,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, Any]:
        """Execute image resize operation.

        Args:
            job: The Job object
            params: Validated ImageResizeParams
            progress_callback: Optional callback to report progress (0-100)

        Returns:
            Result dict with status and task_output
        """
        try:
            from PIL import Image
        except ImportError:
            return {
                "status": "error",
                "error": "Pillow is not installed. Install with: pip install cl_ml_tools[compute]",
            }

        try:
            processed_files = []
            total_files = len(params.input_paths)

            for i, (input_path, output_path) in enumerate(
                zip(params.input_paths, params.output_paths)
            ):
                # Load image
                with Image.open(input_path) as img:
                    # Calculate new size
                    if params.maintain_aspect_ratio:
                        img.thumbnail(
                            (params.width, params.height), Image.Resampling.LANCZOS
                        )
                        resized = img
                    else:
                        resized = img.resize(
                            (params.width, params.height), Image.Resampling.LANCZOS
                        )

                    # Save resized image
                    resized.save(output_path)

                processed_files.append(output_path)

                # Report progress
                if progress_callback:
                    percentage = int((i + 1) / total_files * 100)
                    progress_callback(percentage)

            return {
                "status": "ok",
                "task_output": {
                    "processed_files": processed_files,
                    "dimensions": {"width": params.width, "height": params.height},
                },
            }

        except FileNotFoundError as e:
            return {"status": "error", "error": f"Input file not found: {e}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
