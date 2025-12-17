"""ComputeModule - Abstract base class for compute tasks."""

from abc import ABC, abstractmethod
from typing import Callable, Generic

from .file_storage import JobStorage
from .schema_job import P, Q
from .schema_job_record import JobRecord, JobRecordUpdate


class ComputeModule(ABC, Generic[P, Q]):
    """
    Stateless, template-method based compute module.

    - Params are validated once and passed through
    - run() owns persistence
    - Q contains metadata only
    """

    schema: type[P]

    @property
    @abstractmethod
    def task_type(self) -> str: ...

    def setup(self) -> None:
        """Optional per-execution setup."""
        pass

    @abstractmethod
    async def run(
        self,
        params: P,
        storage: JobStorage,
        progress_callback: Callable[[int], None] | None = None,
    ) -> Q:
        """
        Execute task.

        - May persist data via storage
        - Must return metadata only
        """
        ...

    async def execute(
        self,
        job_record: JobRecord,
        storage: JobStorage,
        progress_callback: Callable[[int], None] | None = None,
    ) -> JobRecordUpdate:
        try:
            params = self.schema.model_validate(job_record.params)

            self.setup()

            output = await self.run(
                params,
                storage,
                progress_callback,
            )

            return JobRecordUpdate(
                status="completed",
                output=output.model_dump(),
                progress=100,
            )

        except FileNotFoundError as exc:
            return JobRecordUpdate(
                status="error",
                error_message=str(exc),
            )

        except Exception as exc:
            return JobRecordUpdate(
                status="error",
                error_message=str(exc),
            )
