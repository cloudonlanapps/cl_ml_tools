"""FileStorage Protocol - interface for file storage operations."""

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class FileStorage(Protocol):
    """Protocol for file storage operations.
    
    Applications must implement this protocol to provide file storage
    functionality. All paths returned by methods are absolute paths.
    
    Example implementation using local filesystem:
    
        class LocalFileStorage:
            def __init__(self, base_dir: str):
                self.base_dir = Path(base_dir)
            
            def create_job_directory(self, job_id: str) -> Path:
                job_dir = self.base_dir / "jobs" / job_id
                (job_dir / "input").mkdir(parents=True, exist_ok=True)
                (job_dir / "output").mkdir(parents=True, exist_ok=True)
                return job_dir
            
            # ... implement other methods
    """

    def create_job_directory(self, job_id: str) -> Path:
        """Create job directory structure with input/output subdirectories.
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Absolute path to the job directory
        """
        ...

    def get_input_path(self, job_id: str) -> Path:
        """Get absolute path to job's input directory.
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Absolute path to the input directory
        """
        ...

    def get_output_path(self, job_id: str) -> Path:
        """Get absolute path to job's output directory.
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Absolute path to the output directory
        """
        ...

    async def save_input_file(self, job_id: str, filename: str, file) -> dict:
        """Save uploaded file to job's input directory.
        
        Args:
            job_id: Unique job identifier
            filename: Target filename
            file: File object (e.g., FastAPI UploadFile)
            
        Returns:
            Dict with file metadata:
            {
                "filename": str,  # Saved filename
                "path": str,      # Absolute path to saved file
                "size": int,      # File size in bytes
                "hash": str       # File hash (e.g., SHA256)
            }
        """
        ...

    def cleanup_job(self, job_id: str) -> bool:
        """Delete job directory and all its files.
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            True if deleted, False if directory didn't exist
        """
        ...
