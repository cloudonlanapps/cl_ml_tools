"""Image conversion task implementation."""

from pathlib import Path
from typing import Callable, Optional, Dict, Any, Type

from cl_ml_tools.common.compute_module import ComputeModule
from cl_ml_tools.common.schemas import Job, BaseJobParams
from .schema import ImageConversionParams


class ImageConversionTask(ComputeModule):
    """Compute module for converting images between formats.

    Converts images to specified format using Pillow.
    """

    @property
    def task_type(self) -> str:
        """Return task type identifier."""
        return "image_conversion"

    def get_schema(self) -> Type[BaseJobParams]:
        """Return the Pydantic params class for this task."""
        return ImageConversionParams

    async def execute(
        self,
        job: Job,
        params: ImageConversionParams,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, Any]:
        """Execute image conversion operation.

        Args:
            job: The Job object
            params: Validated ImageConversionParams
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
                    # Convert mode if necessary for certain formats
                    if params.format.lower() in ("jpg", "jpeg") and img.mode in (
                        "RGBA",
                        "P",
                    ):
                        # Convert to RGB for JPEG (no alpha channel support)
                        img = img.convert("RGB")

                    # Save in target format
                    save_kwargs = {}

                    # Set quality for formats that support it
                    if params.format.lower() in ("jpg", "jpeg", "webp"):
                        save_kwargs["quality"] = params.quality

                    # PNG optimization
                    if params.format.lower() == "png":
                        save_kwargs["optimize"] = True

                    img.save(
                        output_path,
                        format=self._get_pil_format(params.format),
                        **save_kwargs,
                    )

                processed_files.append(output_path)

                # Report progress
                if progress_callback:
                    percentage = int((i + 1) / total_files * 100)
                    progress_callback(percentage)

            return {
                "status": "ok",
                "task_output": {
                    "processed_files": processed_files,
                    "format": params.format,
                    "quality": params.quality,
                },
            }

        except FileNotFoundError as e:
            return {"status": "error", "error": f"Input file not found: {e}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @staticmethod
    def _get_pil_format(format_str: str) -> str:
        """Convert format string to PIL format name."""
        format_map = {
            "jpg": "JPEG",
            "jpeg": "JPEG",
            "png": "PNG",
            "webp": "WEBP",
            "gif": "GIF",
            "bmp": "BMP",
            "tiff": "TIFF",
        }
        return format_map.get(format_str.lower(), format_str.upper())
