from typing import Callable
from uuid import uuid4

from fastapi import UploadFile

from .job_repository import JobRepository
from .job_storage import JobStorage
from .schema_job import BaseJobParams, Job, P, Q, TaskOutput
from .schema_job_record import JobCreatedResponse, JobStatus
from .user import UserLike


async def create_job_from_upload(
    *,
    task_type: str,
    repository: JobRepository,
    file_storage: JobStorage,
    file: UploadFile,
    params_factory: Callable[[str], P],
    output_type: type[Q],  # 👈 binds Q
    priority: int,
    user: UserLike | None,
) -> JobCreatedResponse:
    _ = output_type
    job_id = str(uuid4())

    file_storage.create_directory(job_id)

    if not file.filename:
        raise ValueError("Uploaded file has no filename")

    file_info = await file_storage.save(job_id, f"input/{file.filename}", file)

    params = params_factory(file_info.relative_path)

    job = Job[P, Q](
        job_id=job_id,
        task_type=task_type,
        params=params,
        progress=0,
        status=JobStatus.queued,
    )

    created_by = user.id if user else None
    ok = repository.add_job(
        job.to_record(),
        created_by=created_by,
        priority=priority,
    )

    if not ok:
        raise ValueError("Failed to create job")

    return JobCreatedResponse(job_id=job_id, status=job.status, task_type=task_type)
async def create_job_from_params(
    *,
    task_type: str,
    repository: JobRepository,
    params: BaseJobParams,
    output_type: type[TaskOutput],  # 👈 binds Q
    priority: int,
    user: UserLike | None,
) -> JobCreatedResponse:
    _ = output_type
    job_id = str(uuid4())

    job = Job[BaseJobParams, TaskOutput](
        job_id=job_id,
        task_type=task_type,
        params=params,
        progress=0,
        status=JobStatus.queued,
    )

    created_by = user.id if user else None
    ok = repository.add_job(
        job.to_record(),
        created_by=created_by,
        priority=priority,
    )

    if not ok:
        raise ValueError("Failed to create job")

    return JobCreatedResponse(job_id=job_id, status=job.status, task_type=task_type)
