"""ComputeModule - Abstract base class for compute tasks."""

from abc import ABC, abstractmethod
from typing import Callable, Generic

from .schema_job import P, Q


class ComputeModule(ABC, Generic[P, Q]):
    """
    Abstract base class for all compute tasks.
    All task plugins must extend this class and implement the required methods.
    """

    @property
    @abstractmethod
    def task_type(self) -> str:
        """Return task type identifier."""
        ...

    @abstractmethod
    def get_schema(self) -> type[P]:
        """Return the Pydantic params class for this task."""
        ...

    @abstractmethod
    async def execute(
        self,
        job_id: str,
        params: P,
        progress_callback: Callable[[int], None] | None = None,
    ) -> Q:
        """Execute the task."""
        ...
