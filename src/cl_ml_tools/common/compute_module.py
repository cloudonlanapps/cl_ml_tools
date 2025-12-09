"""ComputeModule - Abstract base class for compute tasks."""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any, Type

from .schemas import BaseJobParams, Job


class ComputeModule(ABC):
    """Abstract base class for all compute tasks.
    
    All task plugins must extend this class and implement the required methods.
    
    Example:
    
        class ImageResizeTask(ComputeModule):
            @property
            def task_type(self) -> str:
                return "image_resize"
            
            def get_schema(self) -> Type[BaseJobParams]:
                return ImageResizeParams
            
            async def execute(self, job, params, progress_callback=None):
                # Perform resize operation
                return {"status": "ok", "task_output": {...}}
    """

    @property
    @abstractmethod
    def task_type(self) -> str:
        """Return task type identifier.
        
        This should be a unique string that identifies this task type
        (e.g., 'image_resize', 'image_conversion', 'video_transcode').
        
        Returns:
            Task type string identifier
        """
        ...

    @abstractmethod
    def get_schema(self) -> Type[BaseJobParams]:
        """Return the Pydantic params class for this task.
        
        The returned class should extend BaseJobParams and define
        any additional parameters required for this task.
        
        Returns:
            Pydantic model class for task parameters
        """
        ...

    @abstractmethod
    async def execute(
        self,
        job: Job,
        params: BaseJobParams,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> Dict[str, Any]:
        """Execute the task.
        
        Args:
            job: The Job object (for access to job_id, task_type if needed)
            params: Validated parameters (subclass of BaseJobParams)
            progress_callback: Optional callback to report progress (0-100)
            
        Returns:
            Result dictionary with one of:
            - Success: {"status": "ok", "task_output": {...}}
            - Failure: {"status": "error", "error": "error message"}
            
        Example:
            async def execute(self, job, params, progress_callback=None):
                try:
                    for i, (inp, out) in enumerate(
                        zip(params.input_paths, params.output_paths)
                    ):
                        # Process file
                        process_file(inp, out)
                        
                        # Report progress
                        if progress_callback:
                            pct = int((i + 1) / len(params.input_paths) * 100)
                            progress_callback(pct)
                    
                    return {
                        "status": "ok",
                        "task_output": {"processed": params.output_paths}
                    }
                except Exception as e:
                    return {"status": "error", "error": str(e)}
        """
        ...
