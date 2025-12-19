"""EXIF extraction route factory."""

from typing import Annotated, Callable

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ...common.job_storage import JobStorage
from ...common.job_creator import create_job_from_upload
from ...common.job_repository import JobRepository
from ...common.schema_job_record import JobCreatedResponse
from ...common.user import UserLike
from .schema import ExifMetadataOutput, ExifMetadataParams


def create_router(
    repository: JobRepository,
    file_storage: JobStorage,
    get_current_user: Callable[[], UserLike | None],
) -> APIRouter:
    """Create FastAPI router for EXIF extraction endpoints."""
    router = APIRouter()

    @router.post("/jobs/exif", response_model=JobCreatedResponse)
    async def create_exif_extraction_job(
        file: Annotated[
            UploadFile, File(description="Media file to extract EXIF metadata from")
        ],
        tags: Annotated[
            str,
            Form(description="Comma-separated EXIF tags to extract (empty for all)"),
        ] = "",
        priority: Annotated[
            int, Form(ge=0, le=10, description="Job priority (0-10)")
        ] = 5,
        user: Annotated[UserLike | None, Depends(get_current_user)] = None,
    ) -> JobCreatedResponse:
        tags_list = (
            [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
        )

        return await create_job_from_upload(
            task_type="exif",
            repository=repository,
            file_storage=file_storage,
            file=file,
            priority=priority,
            user=user,
            output_type=ExifMetadataOutput,
            params_factory=lambda path: ExifMetadataParams(
                input_path=path,
                output_path="output/exif_metadata.json",
                tags=tags_list,
            ),
        )

    _ = create_exif_extraction_job
    return router
