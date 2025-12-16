"""EXIF metadata extraction task implementation."""

import logging
from typing import Callable, TypedDict, override

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from .algo.exif_tool_wrapper import MetadataExtractor
from .schema import ExifMetadata, ExifParams

logger = logging.getLogger(__name__)


class FileResult(TypedDict):
    file_path: str
    status: str
    metadata: dict[str, object]
    error: str | None


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
        try:
            try:
                extractor = MetadataExtractor()
            except RuntimeError as exc:
                logger.error(f"ExifTool initialization failed: {exc}")
                return {
                    "status": "error",
                    "error": (
                        "ExifTool is not installed or not found in PATH. "
                        "Please install ExifTool: https://exiftool.org/"
                    ),
                }

            file_results: list[FileResult] = []
            total_files: int = len(params.input_paths)

            for index, input_path in enumerate(params.input_paths):
                try:
                    if params.tags:
                        raw_metadata = extractor.extract_metadata(input_path, tags=params.tags)
                    else:
                        raw_metadata = extractor.extract_metadata_all(input_path)

                    if raw_metadata:
                        metadata = ExifMetadata.from_raw_metadata(raw_metadata)
                        file_results.append(
                            {
                                "file_path": input_path,
                                "status": "success",
                                "metadata": metadata.model_dump(),
                                "error": None,
                            }
                        )
                    else:
                        logger.warning(f"No EXIF metadata found for {input_path}")
                        file_results.append(
                            {
                                "file_path": input_path,
                                "status": "no_metadata",
                                "metadata": ExifMetadata().model_dump(),
                                "error": None,
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

                except Exception as exc:
                    logger.error(f"Failed to extract metadata from {input_path}: {exc}")
                    file_results.append(
                        {
                            "file_path": input_path,
                            "status": "error",
                            "error": str(exc),
                            "metadata": ExifMetadata().model_dump(),
                        }
                    )

                if progress_callback:
                    progress = int((index + 1) / total_files * 100)
                    progress_callback(progress)

            all_success: bool = all(r["status"] == "success" for r in file_results)
            any_success: bool = any(r["status"] == "success" for r in file_results)

            if all_success or any_success:
                status = "ok"
                if not all_success:
                    success_count = sum(1 for r in file_results if r["status"] == "success")
                    logger.warning(
                        f"Partial success: {success_count}/{total_files} files processed successfully"
                    )
            else:
                return {
                    "status": "error",
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

        except Exception as exc:
            logger.exception(f"Unexpected error in ExifTask: {exc}")
            return {"status": "error", "error": f"Task failed: {exc}"}
