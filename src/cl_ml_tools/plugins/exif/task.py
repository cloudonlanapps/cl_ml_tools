"""EXIF metadata extraction task implementation."""

import json
from typing import Callable, override

from loguru import logger

from ...common.compute_module import ComputeModule
from ...common.job_storage import JobStorage
from .algo.exif_tool_wrapper import MetadataExtractor
from .schema import ExifMetadataOutput, ExifMetadataParams


class ExifTask(ComputeModule[ExifMetadataParams, ExifMetadataOutput]):
    """Compute module for extracting EXIF metadata from a media file."""

    schema: type[ExifMetadataParams] = ExifMetadataParams

    def __init__(self) -> None:
        self._extractor: MetadataExtractor | None = None

    @property
    @override
    def task_type(self) -> str:
        return "exif"

    @override
    def setup(self) -> None:
        if self._extractor is None:
            try:
                self._extractor = MetadataExtractor()
            except RuntimeError as exc:
                logger.error("ExifTool initialization failed", exc_info=exc)
                raise RuntimeError(
                    "ExifTool is not installed or not found in PATH. "
                    + "Please install ExifTool: https://exiftool.org/"
                ) from exc

    @override
    async def run(
        self,
        job_id: str,
        params: ExifMetadataParams,
        storage: JobStorage,
        progress_callback: Callable[[int], None] | None = None,
    ) -> ExifMetadataOutput:
        if not self._extractor:
            raise RuntimeError("ExifTool is not initialized")

        input_path = str(storage.resolve_path(job_id, params.input_path))

        if params.tags:
            raw_metadata = self._extractor.extract_metadata(
                input_path,
                tags=params.tags,
            )
        else:
            raw_metadata = self._extractor.extract_metadata_all(
                input_path,
            )

        if raw_metadata:
            metadata = ExifMetadataOutput.from_raw_metadata(raw_metadata)
        else:
            logger.warning("No EXIF metadata found for %s", params.input_path)
            metadata = ExifMetadataOutput()

        path = storage.allocate_path(
            job_id=job_id,
            relative_path=params.output_path,
        )

        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata.model_dump(), f, indent=2)

        if progress_callback:
            progress_callback(100)

        return metadata
