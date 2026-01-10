"""DINOv2 embedding route factory."""

from typing import Annotated, Callable

from fastapi import APIRouter, Depends, File, Form, UploadFile

from cl_ml_tools.common.user import UserLike

from ...common.job_creator import create_job_from_upload
from ...common.job_repository import JobRepository
from ...common.job_storage import JobStorage
from ...common.schema_job_record import JobCreatedResponse
from .schema import DinoEmbeddingOutput, DinoEmbeddingParams


def create_router(
    repository: JobRepository,
    file_storage: JobStorage,
    get_current_user: Callable[[], UserLike | None],
) -> APIRouter:
    """Create FastAPI router for DINOv2 embedding endpoints."""
    router = APIRouter()

    @router.post("/jobs/dino_embedding", response_model=JobCreatedResponse)
    async def create_dino_embedding_job(
        file: Annotated[
            UploadFile, File(description="Image file for visual similarity embedding")
        ],
        normalize: Annotated[
            bool, Form(description="Whether to L2-normalize the embedding")
        ] = True,
        priority: Annotated[
            int, Form(ge=0, le=10, description="Job priority (0-10)")
        ] = 5,
        user: Annotated[UserLike | None, Depends(get_current_user)] = None,
    ) -> JobCreatedResponse:
        return await create_job_from_upload(
            task_type="dino_embedding",
            repository=repository,
            file_storage=file_storage,
            file=file,
            priority=priority,
            user=user,
            output_type=DinoEmbeddingOutput,
            params_factory=lambda path: DinoEmbeddingParams(
                input_path=path,
                output_path="output/dino_embedding.nb",
                normalize=normalize,
            ),
        )

    _ = create_dino_embedding_job
    return router
