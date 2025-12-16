"""EXIF metadata extraction task implementation."""

import logging
from typing import Callable, override

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from .algo.exif_tool_wrapper import MetadataExtractor
from .schema import ExifMetadata, ExifParams

logger = logging.getLogger(__name__)


class ExifTask(ComputeModule[ExifParams]):
    """Compute module for extracting EXIF metadata from media files."""

    @property
    @override
    def task_type(self) -> str:
        return "exif"

    @override
    def get_schema(self) -> type[BaseJobParams]:
        return ExifParams

    @override
    async def execute(
        self,
        job: Job,
        params: ExifParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        """Extract EXIF metadata from input files.

        Args:
            job: Job instance
            params: ExifParams with input_paths and tags
            progress_callback: Optional callback for progress updates (0-100)

        Returns:
            TaskResult with status and metadata for each file
        """
        try:
            # Initialize ExifTool wrapper
            try:
                extractor = MetadataExtractor()
            except RuntimeError as e:
                logger.error(f"ExifTool initialization failed: {e}")
                return {
                    "status": "error",
                    "error": (
                        "ExifTool is not installed or not found in PATH. "
                        "Please install ExifTool: https://exiftool.org/"
                    ),
                }

            file_results: list[dict] = []
            total_files = len(params.input_paths)

            for index, input_path in enumerate(params.input_paths):
                try:
                    # Extract raw metadata
                    if params.tags:
                        raw_metadata = extractor.extract_metadata(input_path, tags=params.tags)
                    else:
                        # Extract all tags if none specified
                        raw_metadata = extractor.extract_metadata_all(input_path)

                    # Convert to typed model
                    if raw_metadata:
                        metadata = ExifMetadata.from_raw_metadata(raw_metadata)
                        file_results.append(
                            {
                                "file_path": input_path,
                                "status": "success",
                                "metadata": metadata.model_dump(),
                            }
                        )
                    else:
                        # No metadata found (corrupt file or unsupported format)
                        logger.warning(f"No EXIF metadata found for {input_path}")
                        file_results.append(
                            {
                                "file_path": input_path,
                                "status": "no_metadata",
                                "metadata": ExifMetadata().model_dump(),  # Empty metadata
                            }
                        )

                except FileNotFoundError:
                    logger.error(f"File not found: {input_path}")
                    file_results.append(
                        {
                            "file_path": input_path,
                            "status": "error",
                            "error": "File not found",
                            "metadata": ExifMetadata().model_dump(),
                        }
                    )

                except Exception as e:
                    logger.error(f"Failed to extract metadata from {input_path}: {e}")
                    file_results.append(
                        {
                            "file_path": input_path,
                            "status": "error",
                            "error": str(e),
                            "metadata": ExifMetadata().model_dump(),
                        }
                    )

                # Report progress
                if progress_callback:
                    progress = int((index + 1) / total_files * 100)
                    progress_callback(progress)

            # Determine overall status
            all_success = all(r["status"] == "success" for r in file_results)
            any_success = any(r["status"] == "success" for r in file_results)

            if all_success:
                status = "ok"
            elif any_success:
                status = "ok"  # Partial success still returns "ok"
                logger.warning(
                    f"Partial success: {sum(1 for r in file_results if r['status'] == 'success')}"
                    f"/{total_files} files processed successfully"
                )
            else:
                status = "error"
                return {
                    "status": status,
                    "error": "Failed to extract metadata from all files",
                    "task_output": {
                        "files": file_results,
                        "total_files": total_files,
                    },
                }

            return {
                "status": status,
                "task_output": {
                    "files": file_results,
                    "total_files": total_files,
                    "tags_requested": params.tags if params.tags else "all",
                },
            }

        except Exception as e:
            logger.exception(f"Unexpected error in ExifTask: {e}")
            return {"status": "error", "error": f"Task failed: {str(e)}"}
