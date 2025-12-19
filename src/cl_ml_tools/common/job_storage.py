"""
JobStorage Protocol - interface for job-scoped file storage operations.

Design goals:
- Hide internal folder structure
- Support async uploads & streaming
- Support filesystem-bound libraries (numpy, opencv, ffmpeg)
- Keep storage as the single authority over paths
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class JobStorageError(Exception):
    """Base class for storage-related errors."""


class JobDirectoryCreationError(JobStorageError):
    def __init__(self, job_id: str | int):
        self.job_id: str | int = job_id
        super().__init__(f"Failed to create storage directory for job '{job_id}'")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SavedJobFile(BaseModel):
    """Metadata of a saved job file."""

    relative_path: str = Field(
        ...,
        description="Relative path of the saved file within the job storage",
    )
    size: int = Field(
        ...,
        ge=0,
        description="File size in bytes",
    )
    hash: str | None = Field(
        None,
        description="Optional content hash (e.g., SHA256)",
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# File-like abstractions
# ---------------------------------------------------------------------------


class AsyncFileLike(Protocol):
    """Minimal async file-like interface."""

    async def read(self, size: int, /) -> bytes: ...


FileLike = AsyncFileLike | bytes | str | PathLike[str]


# ---------------------------------------------------------------------------
# Storage Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class JobStorage(Protocol):
    """
    Protocol for job-scoped file storage.

    Implementations own:
    - storage root
    - directory layout
    - permissions
    - lifecycle management

    Callers interact ONLY via job_id and relative paths.
    """

    # ---------------------------------------------------------------------
    # Job lifecycle
    # ---------------------------------------------------------------------

    def create_directory(self, job_id: str) -> None:
        """
        Create storage directory for a job.

        Returns:
            True if created or already exists, False otherwise.
        """
        ...

    def remove(self, job_id: str) -> bool:
        """
        Remove all files associated with a job.

        Returns:
            True if removed successfully, False otherwise.
        """
        ...

    # ---------------------------------------------------------------------
    # Writing
    # ---------------------------------------------------------------------

    async def save(
        self,
        job_id: str,
        relative_path: str,
        file: FileLike,
        *,
        mkdirs: bool = True,
    ) -> SavedJobFile:
        """
        Save a file into job storage.

        `file` may be:
        - async file-like object (UploadFile, aiofiles, etc.)
        - bytes
        - existing filename or Path (copied)

        Returns:
            Metadata of the saved file.
        """
        ...

    def allocate_path(
        self,
        job_id: str,
        relative_path: str,
        *,
        mkdirs: bool = True,
    ) -> Path:
        """
        Allocate a filesystem path for writing.

        Intended for libraries that require filenames
        (numpy, opencv, ffmpeg, PIL).

        Storage retains control of layout; caller owns writing.
        """
        ...

    # ---------------------------------------------------------------------
    # Reading / resolving
    # ---------------------------------------------------------------------

    async def open(
        self,
        job_id: str,
        relative_path: str,
    ) -> AsyncFileLike:
        """
        Open a stored file for async reading.
        """
        ...

    def resolve_path(
        self,
        job_id: str,
        relative_path: str | None = None,
    ) -> Path:
        """
        Resolve a job-relative path to an absolute filesystem path.

        This is an intentional abstraction boundary and should only
        be used when required by filesystem-bound libraries.
        """
        ...
