from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue

TaskParamsRecord = dict[str, JsonValue]
TaskOutputRecord = dict[str, JsonValue]


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    error = "error"


class JobRecord(BaseModel):
    """Persisted job representation (DB / wire format)."""

    job_id: str
    task_type: str

    params: TaskParamsRecord
    output: TaskOutputRecord | None = None

    status: JobStatus = JobStatus.queued
    progress: int = Field(0, ge=0, le=100)
    error_message: str | None = None

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class JobRecordUpdate(BaseModel):
    status: JobStatus | None = None
    progress: int | None = Field(default=None, ge=0, le=100)
    output: TaskOutputRecord | None = None
    error_message: str | None = None

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class JobCreatedResponse(BaseModel):
    job_id: str
    status: str
    task_type: str
