"""Worker runtime - orchestrates job execution."""

from typing import List, Optional, Dict

from importlib.metadata import entry_points

from .common.job_repository import JobRepository
from .common.compute_module import ComputeModule
from .common.schemas import Job


def get_task_registry() -> Dict[str, ComputeModule]:
    """Dynamically load all tasks from entry points.

    Discovers tasks from [project.entry-points."cl_ml_tools.tasks"]
    in pyproject.toml.

    Returns:
        Dict mapping task_type -> ComputeModule instance

    Raises:
        RuntimeError: If a plugin fails to load (missing dependency, etc.)
    """
    registry = {}
    eps = entry_points(group="cl_ml_tools.tasks")

    for ep in eps:
        try:
            task_class = ep.load()
            task = task_class()
            registry[task.task_type] = task
        except Exception as e:
            # Plugin dependency missing = exception (fail fast)
            raise RuntimeError(f"Failed to load task '{ep.name}': {e}")

    return registry


class Worker:
    """Worker runtime that orchestrates job execution.

    Responsibilities:
    - Maintains task registry (auto-discovered from entry points)
    - Fetches jobs from repository (atomic claim prevents race conditions)
    - Validates task_types against registry before fetching
    - Dispatches jobs to appropriate ComputeModule
    - Handles errors and updates job status

    Example:
        repository = SQLiteJobRepository("./jobs.db")
        worker = Worker(repository)

        # Process jobs forever
        while True:
            if not await worker.run_once():
                await asyncio.sleep(1.0)
    """

    def __init__(
        self,
        repository: JobRepository,
        task_registry: Optional[Dict[str, ComputeModule]] = None,
    ):
        """Initialize worker.

        Args:
            repository: JobRepository implementation
            task_registry: Optional custom registry. If None, auto-discovers from entry points.
        """
        self.repository = repository
        self.task_registry = (
            task_registry if task_registry is not None else get_task_registry()
        )

    def get_supported_task_types(self) -> List[str]:
        """Return list of task types this worker can handle.

        Returns:
            List of task type identifiers
        """
        return list(self.task_registry.keys())

    async def run_once(self, task_types: Optional[List[str]] = None) -> bool:
        """Process one job and return.

        Args:
            task_types: List of task types to process.
                        If None, uses all registered task types.

        Returns:
            True if job was processed, False if no jobs available.
        """
        # Validate: only request jobs we can handle
        if task_types is None:
            valid_types = self.get_supported_task_types()
        else:
            valid_types = [t for t in task_types if t in self.task_registry]

        if not valid_types:
            return False  # No handlers for any requested types

        # Fetch job (atomic claim - no race condition)
        # fetch_next_job() atomically finds AND sets status="processing"
        job = self.repository.fetch_next_job(valid_types)
        if not job:
            return False

        # Get task handler - guaranteed to exist after validation
        task = self.task_registry[job.task_type]

        # Execute task
        await self._execute_task(job, task)

        return True

    async def _execute_task(self, job: Job, task: ComputeModule) -> None:
        """Execute a task and update job status.

        Args:
            job: Job to execute
            task: ComputeModule to use for execution
        """
        try:
            # Parse params using task's schema
            params_class = task.get_schema()
            params = params_class(**job.params)

            # Progress callback updates repository
            def progress_callback(pct: int) -> None:
                self.repository.update_job(job.job_id, progress=min(99, pct))

            # Execute task
            result = await task.execute(job, params, progress_callback)

            # Update status based on result
            if result.get("status") == "ok":
                self.repository.update_job(
                    job.job_id,
                    status="completed",
                    progress=100,
                    task_output=result.get("task_output"),
                )
            else:
                self.repository.update_job(
                    job.job_id,
                    status="error",
                    error_message=result.get("error", "Unknown error"),
                )
        except Exception as e:
            self.repository.update_job(job.job_id, status="error", error_message=str(e))
