"""EXIF extraction route factory."""

from typing import Annotated, Callable, Literal, Protocol, TypedDict
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ...common.file_storage import FileStorage
from ...common.job_repository import JobRepository
from ...common.schemas import Job


class UserLike(Protocol):
    id: str | None


class JobCreatedResponse(TypedDict):
    job_id: str
    status: Literal["queued"]


def create_router(
    repository: JobRepository,
    file_storage: FileStorage,
    get_current_user: Callable[[], UserLike | None],
) -> APIRouter:
    """Create FastAPI router for EXIF extraction endpoints.

    Args:
        repository: Job repository for persistence
        file_storage: File storage for managing uploaded files
        get_current_user: Dependency for getting current user

    Returns:
        Configured APIRouter with EXIF extraction endpoint
    """
    router = APIRouter()

    @router.post("/jobs/exif", response_model=JobCreatedResponse)
    async def create_exif_extraction_job(
        file: Annotated[UploadFile, File(description="Media file to extract EXIF metadata from")],
        tags: Annotated[str, Form(description="Comma-separated EXIF tags to extract (empty for all)")] = "",
        priority: Annotated[int, Form(ge=0, le=10, description="Job priority (0-10)")] = 5,
        user: Annotated[UserLike | None, Depends(get_current_user)] = None,
    ) -> JobCreatedResponse:
        """Create a new EXIF metadata extraction job.

        Args:
            file: Uploaded media file
            tags: Comma-separated list of EXIF tags to extract (e.g., "Make,Model,DateTimeOriginal")
                  Leave empty to extract all available tags
            priority: Job priority (0-10, higher is more priority)
            user: Current user (injected by dependency)

        Returns:
            JobCreatedResponse with job_id and status
        """
        job_id = str(uuid4())

        # Create job directory
        _ = file_storage.create_job_directory(job_id)

        if not file.filename:
            raise ValueError("Uploaded file has no filename")

        filename: str = file.filename

        # Save uploaded file
        file_info = await file_storage.save_input_file(job_id, filename, file)
        input_path = file_info["path"]

        # Parse tags (comma-separated string to list)
        tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []

        # Create job (no output_paths needed for EXIF extraction)
        job = Job(
            job_id=job_id,
            task_type="exif",
            params={
                "input_paths": [input_path],
                "output_paths": [],  # EXIF extraction doesn't produce output files
                "tags": tags_list,
            },
        )

        created_by = user.id if user is not None else None
        _ = repository.add_job(job, created_by=created_by, priority=priority)

        return {
            "job_id": job_id,
            "status": "queued",
        }

    # Mark function as used (accessed via FastAPI decorator)
    _ = create_exif_extraction_job

    return router
