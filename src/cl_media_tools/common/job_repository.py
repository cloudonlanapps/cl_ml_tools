"""JobRepository Protocol - interface for job persistence."""

from typing import Protocol, Optional, List, runtime_checkable

from .schemas import Job


@runtime_checkable
class JobRepository(Protocol):
    """Protocol for job persistence operations.
    
    Applications must implement this protocol to provide job storage
    functionality. The implementation should handle the underlying
    database operations.
    
    IMPORTANT: The `fetch_next_job` method MUST be atomic to prevent
    race conditions when multiple workers are running.
    
    Example implementation using SQLite:
    
        class SQLiteJobRepository:
            def __init__(self, db_path: str):
                self.conn = sqlite3.connect(db_path)
            
            def fetch_next_job(self, task_types: List[str]) -> Optional[Job]:
                # Use BEGIN IMMEDIATE for atomic claim
                cursor = self.conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                try:
                    # Find and claim job atomically
                    cursor.execute("SELECT * FROM jobs WHERE status = 'queued' ...")
                    row = cursor.fetchone()
                    if row:
                        cursor.execute("UPDATE jobs SET status = 'processing' ...")
                        self.conn.commit()
                        return Job(**row)
                    self.conn.rollback()
                    return None
                except:
                    self.conn.rollback()
                    raise
    """

    def add_job(
        self,
        job: Job,
        created_by: Optional[str] = None,
        priority: Optional[int] = None
    ) -> str:
        """Save job to database.
        
        Args:
            job: Job object to save
            created_by: Optional user identifier who created the job
            priority: Optional job priority (higher = more urgent)
            
        Returns:
            The job_id of the saved job
        """
        ...

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID.
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Job object if found, None otherwise
        """
        ...

    def update_job(self, job_id: str, **kwargs) -> bool:
        """Update job fields.
        
        Commonly updated fields:
        - status: "queued" | "processing" | "completed" | "error"
        - progress: 0-100
        - task_output: dict with task results
        - error_message: string with error details
        
        Args:
            job_id: Unique job identifier
            **kwargs: Fields to update
            
        Returns:
            True if job was updated, False if job not found
        """
        ...

    def fetch_next_job(self, task_types: List[str]) -> Optional[Job]:
        """Atomically find and claim the next queued job.
        
        This method MUST be atomic to prevent race conditions when
        multiple workers are running. The implementation should:
        
        1. Find job with status="queued" AND task_type in task_types
        2. Atomically update status to "processing"
        3. Return the claimed job
        
        Database-specific implementations:
        - PostgreSQL: UPDATE ... WHERE ... RETURNING *
        - MySQL: SELECT ... FOR UPDATE SKIP LOCKED + UPDATE
        - SQLite: BEGIN IMMEDIATE + SELECT + UPDATE + COMMIT
        
        Args:
            task_types: List of task types to process
            
        Returns:
            Job object with status="processing" if found, None otherwise
        """
        ...

    def delete_job(self, job_id: str) -> bool:
        """Delete job from database.
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            True if job was deleted, False if job not found
        """
        ...
