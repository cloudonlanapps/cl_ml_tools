"""Face detection route factory."""

from typing import Annotated, Callable

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ...common.job_storage import JobStorage
from ...common.job_creator import create_job_from_upload
from ...common.job_repository import JobRepository
from ...common.schema_job_record import JobCreatedResponse
from ...common.user import UserLike
from .schema import FaceDetectionOutput, FaceDetectionParams


def create_router(
    repository: JobRepository,
    file_storage: JobStorage,
    get_current_user: Callable[[], UserLike | None],
) -> APIRouter:
    """Create FastAPI router for face detection endpoints."""
    router = APIRouter()

    @router.post("/jobs/face_detection", response_model=JobCreatedResponse)
    async def create_face_detection_job(
        file: Annotated[UploadFile, File(description="Image file to detect faces in")],
        confidence_threshold: Annotated[
            float,
            Form(ge=0.0, le=1.0, description="Minimum confidence threshold (0.0-1.0)"),
        ] = 0.7,
        nms_threshold: Annotated[
            float,
            Form(
                ge=0.0,
                le=1.0,
                description="NMS threshold for overlapping boxes (0.0-1.0)",
            ),
        ] = 0.4,
        priority: Annotated[
            int, Form(ge=0, le=10, description="Job priority (0-10)")
        ] = 5,
        user: Annotated[UserLike | None, Depends(get_current_user)] = None,
    ) -> JobCreatedResponse:
        return await create_job_from_upload(
            task_type="face_detection",
            repository=repository,
            file_storage=file_storage,
            file=file,
            priority=priority,
            user=user,
            output_type=FaceDetectionOutput,
            params_factory=lambda path: FaceDetectionParams(
                input_path=path,
                output_path="output/face_detection.json",
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
            ),
        )

    _ = create_face_detection_job
    return router
