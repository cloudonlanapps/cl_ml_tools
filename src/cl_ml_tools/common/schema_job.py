from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from .schema_job_record import JobRecord, JobRecordUpdate


class BaseJobParams(BaseModel):
    input_path: str = Field(description="path to the input file")
    output_path: str = Field(description="path to the output file")


class TaskOutput(BaseModel):
    pass


P = TypeVar("P", bound=BaseJobParams)
Q = TypeVar("Q", bound=TaskOutput)


class Job(BaseModel, Generic[P, Q]):
    """Runtime, strongly-typed job."""

    job_id: str
    task_type: str

    params: P
    output: Q | None = None

    status: str = "queued"
    progress: int = Field(0, ge=0, le=100)
    error_message: str | None = None

    model_config: ClassVar[ConfigDict] = ConfigDict(from_attributes=True)

    # ---------------------------
    # Runtime → Persisted
    # ---------------------------
    def to_record(self) -> JobRecord:
        return JobRecord(
            job_id=self.job_id,
            task_type=self.task_type,
            params=self.params.model_dump(),
            output=self.output.model_dump() if self.output is not None else None,
            status=self.status,
            progress=self.progress,
            error_message=self.error_message,
        )

    # ---------------------------
    # Persisted → Runtime
    # ---------------------------
    @classmethod
    def from_record(
        cls,
        record: JobRecord,
        params_cls: type[P],
        output_cls: type[Q],
    ) -> "Job[P, Q]":
        return cls(
            job_id=record.job_id,
            task_type=record.task_type,
            params=params_cls.model_validate(record.params),
            output=(
                output_cls.model_validate(record.output) if record.output is not None else None
            ),
            status=record.status,
            progress=record.progress,
            error_message=record.error_message,
        )

    # ---------------------------
    # Apply partial DB update
    # ---------------------------
    def from_record_update(self, update: JobRecordUpdate) -> "Job[P, Q]":
        return self.model_copy(update=update.model_dump(exclude_none=True))
