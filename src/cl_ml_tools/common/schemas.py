"""Pydantic schemas for job parameters and data structures."""

from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator, model_validator


class BaseJobParams(BaseModel):
    """Base parameters for all compute tasks.
    
    All task-specific parameter classes should extend this.
    
    Attributes:
        input_paths: List of absolute paths to input files
        output_paths: List of absolute paths for output files
    """
    
    input_paths: List[str] = Field(
        default_factory=list, 
        description="List of absolute paths to input files"
    )
    output_paths: List[str] = Field(
        default_factory=list, 
        description="List of absolute paths for output files"
    )

    @field_validator("output_paths")
    @classmethod
    def validate_output_paths_unique(cls, v: List[str]) -> List[str]:
        """Ensure output paths are unique."""
        if len(v) != len(set(v)):
            raise ValueError("Output paths must be unique")
        return v

    @model_validator(mode='after')
    def validate_paths_length(self) -> "BaseJobParams":
        """Ensure output paths match input paths count."""
        if self.output_paths and self.input_paths:
            if len(self.output_paths) != len(self.input_paths):
                raise ValueError("Number of output paths must match number of input paths")
        return self


class Job(BaseModel):
    """Job data structure for JobRepository protocol.
    
    This is the structure used to represent jobs in the system.
    Applications implement JobRepository to persist this in their database.
    
    Attributes:
        job_id: Unique job identifier (UUID)
        task_type: Type of task (e.g., 'image_resize')
        params: Task parameters as dictionary
        status: Job status (queued, processing, completed, error)
        progress: Progress percentage (0-100)
        task_output: Task output results
        error_message: Error message if failed
    """
    
    job_id: str = Field(..., description="Unique job identifier (UUID)")
    task_type: str = Field(..., description="Type of task (e.g., 'image_resize')")
    params: Dict[str, Any] = Field(..., description="Task parameters as dictionary")
    status: str = Field(default="queued", description="Job status")
    progress: int = Field(default=0, ge=0, le=100, description="Progress percentage")
    task_output: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Task output results"
    )
    error_message: Optional[str] = Field(
        default=None, 
        description="Error message if failed"
    )

    model_config = {"from_attributes": True}
