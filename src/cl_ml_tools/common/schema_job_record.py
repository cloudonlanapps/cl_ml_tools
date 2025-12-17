from collections.abc import Mapping
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

TaskOutputRecord = Mapping[str, object]
TaskParamsRecord = Mapping[str, object]


class JobRecord(BaseModel):
    """Persisted job representation (DB / wire format)."""

    job_id: str
    task_type: str

    params: TaskParamsRecord
    output: TaskOutputRecord | None = None

    status: str = "queued"
    progress: int = Field(0, ge=0, le=100)
    error_message: str | None = None

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class JobRecordUpdate(BaseModel):
    status: str | None = None
    progress: int | None = Field(default=None, ge=0, le=100)
    output: TaskOutputRecord | None = None
    error_message: str | None = None

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
