"""Pydantic schemas for job parameters and data structures."""

from collections.abc import Mapping, Sequence
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ─────────────────────────────────────────────────────────────
# Base job params
# ─────────────────────────────────────────────────────────────


class BaseJobParams(BaseModel):
    """Base parameters for all compute tasks.

    All task-specific parameter classes should extend this.
    """

    input_paths: list[str] = Field(
        default_factory=list,
        description="List of absolute paths to input files",
    )
    output_paths: list[str] = Field(
        default_factory=list,
        description="List of absolute paths for output files",
    )

    @field_validator("output_paths")
    @classmethod
    def validate_output_paths_unique(cls, v: Sequence[str]) -> Sequence[str]:
        """Ensure output paths are unique."""
        if len(v) != len(set(v)):
            raise ValueError("Output paths must be unique")
        return v

    @model_validator(mode="after")
    def validate_paths_length(self) -> "BaseJobParams":
        """Ensure output paths match input paths count."""
        if self.output_paths and self.input_paths:
            if len(self.output_paths) != len(self.input_paths):
                raise ValueError("Number of output paths must match number of input paths")
        return self


# ─────────────────────────────────────────────────────────────
# Job task output (loosely structured but not Any)
# ─────────────────────────────────────────────────────────────

TaskOutput = Mapping[str, object]


# ─────────────────────────────────────────────────────────────
# Task execution result
# ─────────────────────────────────────────────────────────────
class TaskResult(BaseModel):
    """Result returned by ComputeModule.execute()."""

    status: str
    task_output: TaskOutput | None = None
    error: str | None = None

    model_config = ConfigDict(extra="forbid")


# ─────────────────────────────────────────────────────────────
# Job model
# ─────────────────────────────────────────────────────────────


class Job(BaseModel):
    """Job data structure for JobRepository protocol.

    This is the structure used to represent jobs in the system.
    """

    job_id: str = Field(..., description="Unique job identifier (UUID)")
    task_type: str = Field(..., description="Type of task (e.g., 'image_thumbnail')")

    # Params are stored as raw data, validated separately via BaseJobParams
    params: Mapping[str, object] = Field(..., description="Task parameters as dictionary")

    status: str = Field(default="queued", description="Job status")
    progress: int = Field(default=0, ge=0, le=100, description="Progress percentage")

    task_output: TaskOutput | None = Field(default=None, description="Task output results")

    error_message: str | None = Field(default=None, description="Error message if failed")

    model_config: ClassVar[ConfigDict] = {
        "from_attributes": True,
    }
