"""Image conversion route factory."""

from pathlib import Path
from typing import Annotated, Callable, Literal

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ...common.file_storage import JobStorage
from ...common.job_creator import create_job_from_upload
from ...common.job_repository import JobRepository
from ...common.schema_job_record import JobCreatedResponse
from ...common.user import UserLike
from .schema import ImageConversionOutput, ImageConversionParams


def create_router(
    repository: JobRepository,
    file_storage: JobStorage,
    get_current_user: Callable[[], UserLike | None],
) -> APIRouter:
    router = APIRouter()

    @router.post("/jobs/image_conversion", response_model=JobCreatedResponse)
    async def create_conversion_job(
        file: Annotated[UploadFile, File(description="Image file to convert")],
        format: Annotated[
            Literal["png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"],
            Form(description="Target format"),
        ],
        quality: Annotated[int, Form(ge=1, le=100, description="Output quality (1-100)")] = 85,
        priority: Annotated[int, Form(ge=0, le=10, description="Job priority (0-10)")] = 5,
        user: Annotated[UserLike | None, Depends(get_current_user)] = None,
    ) -> JobCreatedResponse:
        output_ext = "jpg" if format == "jpeg" else format

        return await create_job_from_upload(
            task_type="image_conversion",
            repository=repository,
            file_storage=file_storage,
            file=file,
            priority=priority,
            user=user,
            output_type=ImageConversionOutput,
            params_factory=lambda path: ImageConversionParams(
                input_path=path,
                output_path=f"output/converted_{Path(path).stem}.{output_ext}",
                format=format,
                quality=quality,
            ),
        )

    _ = create_conversion_job
    return router
