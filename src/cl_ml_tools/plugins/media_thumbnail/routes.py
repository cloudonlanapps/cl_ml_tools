"""Media thumbnail route factory."""

from typing import Annotated, Callable

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ...common.job_creator import create_job_from_upload
from ...common.job_repository import JobRepository
from ...common.job_storage import JobStorage
from ...common.schema_job_record import JobCreatedResponse
from ...common.user import UserLike
from .schema import MediaThumbnailOutput, MediaThumbnailParams


def create_router(
    repository: JobRepository,
    file_storage: JobStorage,
    get_current_user: Callable[[], UserLike | None],
) -> APIRouter:
    """Create router with injected dependencies."""
    router = APIRouter()

    @router.post("/jobs/media_thumbnail", response_model=JobCreatedResponse)
    async def create_thumbnail_job(
        file: Annotated[
            UploadFile, File(description="Media file to thumbnail (image or video)")
        ],
        width: Annotated[
            int | None, Form(gt=0, description="Target width in pixels")
        ] = None,
        height: Annotated[
            int | None, Form(gt=0, description="Target height in pixels")
        ] = None,
        maintain_aspect_ratio: Annotated[
            bool, Form(description="Maintain aspect ratio")
        ] = True,
        priority: Annotated[
            int, Form(ge=0, le=10, description="Job priority (0-10)")
        ] = 5,
        user: Annotated[UserLike | None, Depends(get_current_user)] = None,
    ) -> JobCreatedResponse:
        return await create_job_from_upload(
            task_type="media_thumbnail",
            repository=repository,
            file_storage=file_storage,
            file=file,
            priority=priority,
            user=user,
            output_type=MediaThumbnailOutput,
            params_factory=lambda path: MediaThumbnailParams(
                input_path=path,
                output_path="output/thumbnail.jpg",
                width=width,
                height=height,
                maintain_aspect_ratio=maintain_aspect_ratio,
            ),
        )

    _ = create_thumbnail_job
    return router
