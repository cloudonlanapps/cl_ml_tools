"""FastAPI routes for HLS streaming conversion plugin."""

import json
from typing import Annotated, Callable, Protocol, cast
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
    """Create router with injected dependencies."""
    router = APIRouter()

    @router.post("/jobs/hls_streaming")
    async def create_hls_job(
        file: Annotated[UploadFile, File(description="Video file to convert")],
        variants: Annotated[
            str,
            Form(
                description="JSON array of variants: [{resolution:720,bitrate:3500}]"
            ),
        ] = '[{"resolution":720,"bitrate":3500},{"resolution":480,"bitrate":1500}]',
        include_original: Annotated[
            bool, Form(description="Include original quality")
        ] = False,
        priority: Annotated[
            int, Form(ge=0, le=10, description="Job priority (0-10)")
        ] = 5,
        user: Annotated[UserLike | None, Depends(get_current_user)] = None,
    ) -> dict[str, str]:
        """Create an HLS streaming conversion job."""
        job_id = str(uuid4())

        if not file.filename:
            raise ValueError("Uploaded file has no filename")

        filename: str = file.filename

        # Create job directory and save uploaded file
        _ = file_storage.create_job_directory(job_id)
        file_info = await file_storage.save_input_file(job_id, filename, file)
        input_path = file_info["path"]

        # Output directory for HLS files
        output_dir = str(file_storage.get_output_path(job_id))

        # Parse variants JSON
        variants_list = cast(list[dict[str, int | None]], json.loads(variants))

        # Create job
        job = Job(
            job_id=job_id,
            task_type="hls_streaming",
            params={
                "input_paths": [input_path],
                "output_paths": [output_dir],
                "variants": variants_list,
                "include_original": include_original,
            },
        )

        # Save to repository
        created_by = user.id if user is not None else None
        _ = repository.add_job(job, created_by=created_by, priority=priority)

        return {"job_id": job_id, "status": "queued"}

    _ = create_hls_job
    return router
