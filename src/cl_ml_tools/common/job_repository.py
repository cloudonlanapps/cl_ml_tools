"""JobRepository Protocol - interface for job persistence."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .schema_job_record import JobRecord, JobRecordUpdate


@runtime_checkable
class JobRepository(Protocol):
    """Protocol for job persistence operations.

    Applications must implement this protocol to provide job storage
    functionality. Implementations must ensure atomic job claiming.
    """

    def add_job(
        self,
        job: JobRecord,
        created_by: str | None = None,
        priority: int | None = None,
    ) -> str:
        """Save job to database.

        Returns:
            The job_id of the saved job
        """
        ...

    def get_job(self, job_id: str) -> JobRecord | None:
        """Get job by ID."""
        ...

    def update_job(
        self,
        job_id: str,
        updates: JobRecordUpdate,
    ) -> bool:
        """Update job fields.

        Args:
            job_id: Unique job identifier
            updates: Typed job update payload

        Returns:
            True if job was updated, False if job not found
        """
        ...

    def fetch_next_job(
        self,
        task_types: Sequence[str],
    ) -> JobRecord | None:
        """Atomically find and claim the next queued job.

        Implementations MUST ensure this operation is atomic.
        """
        ...

    def delete_job(self, job_id: str) -> bool:
        """Delete job from database."""
        ...
