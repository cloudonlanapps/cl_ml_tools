"""Hash computation route factory."""

from typing import Annotated, Callable, Literal

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ...common.job_storage import JobStorage
from ...common.job_creator import create_job_from_upload
from ...common.job_repository import JobRepository
from ...common.schema_job_record import JobCreatedResponse
from ...common.user import UserLike
from .schema import HashOutput, HashParams


def create_router(
    repository: JobRepository,
    file_storage: JobStorage,
    get_current_user: Callable[[], UserLike | None],
) -> APIRouter:
    """Create router with injected dependencies."""
    router = APIRouter()

    @router.post("/jobs/hash", response_model=JobCreatedResponse)
    async def create_hash_job(
        file: Annotated[UploadFile, File(description="File to hash")],
        algorithm: Annotated[
            Literal["sha512", "md5"],
            Form(description="Hash algorithm to use (sha512 or md5)"),
        ] = "sha512",
        priority: Annotated[
            int, Form(ge=0, le=10, description="Job priority (0-10)")
        ] = 5,
        user: Annotated[UserLike | None, Depends(get_current_user)] = None,
    ) -> JobCreatedResponse:
        return await create_job_from_upload(
            task_type="hash",
            repository=repository,
            file_storage=file_storage,
            file=file,
            priority=priority,
            user=user,
            output_type=HashOutput,
            params_factory=lambda path: HashParams(
                input_path=path,
                output_path="output/hash.json",
                algorithm=algorithm,
            ),
        )

    _ = create_hash_job
    return router
