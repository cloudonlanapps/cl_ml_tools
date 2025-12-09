"""Image resize route factory."""

from typing import Callable
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, Form, Depends

from cl_ml_tools.common.job_repository import JobRepository
from cl_ml_tools.common.file_storage import FileStorage
from cl_ml_tools.common.schemas import Job


def create_router(
    repository: JobRepository, file_storage: FileStorage, get_current_user: Callable
) -> APIRouter:
    """Create router with injected dependencies.

    Args:
        repository: JobRepository implementation
        file_storage: FileStorage implementation
        get_current_user: Callable that returns current user (for auth)

    Returns:
        Configured APIRouter with image resize endpoint
    """
    router = APIRouter()

    @router.post("/jobs/image_resize")
    async def create_resize_job(
        file: UploadFile = File(..., description="Image file to resize"),
        width: int = Form(..., gt=0, description="Target width in pixels"),
        height: int = Form(..., gt=0, description="Target height in pixels"),
        maintain_aspect_ratio: bool = Form(False, description="Maintain aspect ratio"),
        priority: int = Form(5, ge=0, le=10, description="Job priority (0-10)"),
        user=Depends(get_current_user),
    ):
        """Create an image resize job.

        Upload an image and specify target dimensions. The job will be queued
        for processing by a worker.

        Returns:
            job_id: Unique identifier for the created job
            status: Initial job status ("queued")
        """
        job_id = str(uuid4())

        # Create job directory and save uploaded file
        file_storage.create_job_directory(job_id)
        file_info = await file_storage.save_input_file(job_id, file.filename, file)

        # Generate output path
        input_path = file_info["path"]
        output_filename = f"resized_{file.filename}"
        output_path = str(file_storage.get_output_path(job_id) / output_filename)

        # Create job
        job = Job(
            job_id=job_id,
            task_type="image_resize",
            params={
                "input_paths": [input_path],
                "output_paths": [output_path],
                "width": width,
                "height": height,
                "maintain_aspect_ratio": maintain_aspect_ratio,
            },
        )

        # Save to repository
        created_by = user.id if user and hasattr(user, "id") else None
        repository.add_job(job, created_by=created_by, priority=priority)

        return {"job_id": job_id, "status": "queued"}

    return router
