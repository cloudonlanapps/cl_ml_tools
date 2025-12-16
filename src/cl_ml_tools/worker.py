"""Worker runtime - orchestrates job execution."""

from importlib.metadata import entry_points
from typing import cast

from .common.compute_module import ComputeModule
from .common.job_repository import JobRepository, JobUpdate
from .common.schemas import BaseJobParams, Job


def get_task_registry() -> dict[str, ComputeModule[BaseJobParams]]:
    """Dynamically load all tasks from entry points.

    Discovers tasks from [project.entry-points."cl_ml_tools.tasks"]
    in pyproject.toml.

    Returns:
        Dict mapping task_type -> ComputeModule instance

    Raises:
        RuntimeError: If a plugin fails to load (missing dependency, etc.)
    """
    registry: dict[str, ComputeModule[BaseJobParams]] = {}
    eps = entry_points(group="cl_ml_tools.tasks")

    for ep in eps:
        try:
            task_class = cast(type[ComputeModule[BaseJobParams]], ep.load())
            task: ComputeModule[BaseJobParams] = task_class()
            task_type: str = task.task_type
            registry[task_type] = task
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
        task_registry: dict[str, ComputeModule[BaseJobParams]] | None = None,
    ):
        """Initialize worker.

        Args:
            repository: JobRepository implementation
            task_registry: Optional custom registry. If None, auto-discovers from entry points.
        """
        self.repository: JobRepository = repository
        self.task_registry: dict[str, ComputeModule[BaseJobParams]] = (
            task_registry if task_registry is not None else get_task_registry()
        )

    def get_supported_task_types(self) -> list[str]:
        """Return list of task types this worker can handle.

        Returns:
            List of task type identifiers
        """
        return list(self.task_registry.keys())

    async def run_once(self, task_types: list[str] | None = None) -> bool:
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

    async def _execute_task(self, job: Job, task: ComputeModule[BaseJobParams]) -> None:
        """Execute a task and update job status.

        Args:
            job: Job to execute
            task: ComputeModule to use for execution
        """
        try:
            # Parse params using task's schema
            params_class = task.get_schema()
            # Use Pydantic's model_validate to handle Mapping[str, object] -> BaseJobParams
            params = params_class.model_validate(job.params)

            # Progress callback updates repository
            def progress_callback(pct: int) -> None:
                _ = self.repository.update_job(job.job_id, {"progress": min(99, pct)})

            # Execute task
            result = await task.execute(job, params, progress_callback)

            # Update status based on result
            if result.status == "ok":
                updates: JobUpdate = {
                    "status": "completed",
                    "progress": 100,
                }
                task_output = result.task_output
                if task_output is not None:
                    updates["task_output"] = dict(task_output)
                _ = self.repository.update_job(job.job_id, updates)
            else:
                error_msg = result.error
                _ = self.repository.update_job(
                    job.job_id,
                    {
                        "status": "error",
                        "error_message": error_msg if error_msg is not None else "Unknown error",
                    },
                )
        except Exception as e:
            _ = self.repository.update_job(job.job_id, {"status": "error", "error_message": str(e)})
