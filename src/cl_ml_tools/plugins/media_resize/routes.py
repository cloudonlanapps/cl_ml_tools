"""Media resize route factory."""

from typing import Annotated, Callable, Protocol
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ...common.file_storage import FileStorage
from ...common.job_repository import JobRepository
from ...common.schemas import Job


class UserLike(Protocol):
    """Protocol for user objects returned by authentication."""

    id: str | None


def create_router(
    repository: JobRepository,
    file_storage: FileStorage,
    get_current_user: Callable[[], UserLike | None],
) -> APIRouter:
    """Create router with injected dependencies.

    Args:
        repository: JobRepository implementation
        file_storage: FileStorage implementation
        get_current_user: Callable that returns current user (for auth)

    Returns:
        Configured APIRouter with media resize endpoint
    """
    router = APIRouter()

    @router.post("/jobs/media_resize")
    async def create_resize_job(
        file: Annotated[UploadFile, File(description="Media file to resize (image or video)")],
        width: Annotated[int, Form(gt=0, description="Target width in pixels")],
        height: Annotated[int, Form(gt=0, description="Target height in pixels")],
        maintain_aspect_ratio: Annotated[bool, Form(description="Maintain aspect ratio")] = False,
        priority: Annotated[int, Form(ge=0, le=10, description="Job priority (0-10)")] = 5,
        user: Annotated[UserLike | None, Depends(get_current_user)] = None,
    ):
        """Create a media resize job.

        Upload a media file (image or video) and specify target dimensions.
        The job will be queued for processing by a worker.

        Returns:
            job_id: Unique identifier for the created job
            status: Initial job status ("queued")
        """
        job_id = str(uuid4())

        if not file.filename:
            raise ValueError("Uploaded file has no filename")

        filename: str = file.filename

        # Create job directory and save uploaded file
        _ = file_storage.create_job_directory(job_id)
        file_info = await file_storage.save_input_file(job_id, filename, file)

        # Generate output path
        input_path = file_info["path"]
        output_filename = f"resized_{filename}"
        output_path = str(file_storage.get_output_path(job_id) / output_filename)

        # Create job
        job = Job(
            job_id=job_id,
            task_type="media_resize",
            params={
                "input_paths": [input_path],
                "output_paths": [output_path],
                "width": width,
                "height": height,
                "maintain_aspect_ratio": maintain_aspect_ratio,
            },
        )

        # Save to repository
        created_by = user.id if user is not None else None
        _ = repository.add_job(job, created_by=created_by, priority=priority)

        return {"job_id": job_id, "status": "queued"}

    # Mark function as used (accessed via FastAPI decorator)
    _ = create_resize_job

    return router
