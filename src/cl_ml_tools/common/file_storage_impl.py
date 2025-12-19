from __future__ import annotations

import hashlib
import shutil
from os import PathLike
from pathlib import Path
from typing import Final, override

import aiofiles

from .job_storage import (
    AsyncFileLike,
    FileLike,
    JobDirectoryCreationError,
    JobStorage,
    SavedJobFile,
)


class LocalFileStorage(JobStorage):
    """
    Local filesystem implementation of JobStorage.

    Layout:
        base_dir/
            <job_id>/
                <relative_path>
    """

    _CHUNK_SIZE: Final[int] = 1024 * 1024  # 1 MB

    def __init__(self, base_dir: str | PathLike[str]):
        self._base_dir: Path = Path(base_dir).expanduser().resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _job_dir(self, job_id: str) -> Path:
        return self._base_dir / job_id

    def _safe_path(self, job_id: str, relative_path: str | None = None) -> Path:
        """
        Resolve and validate a job-relative path.
        Prevents path traversal.
        """
        base = self._job_dir(job_id)

        path = base if relative_path is None else (base / relative_path)
        resolved = path.resolve()

        if base not in resolved.parents and resolved != base:
            raise ValueError("Invalid relative path (path traversal detected)")

        return resolved

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------
    @override
    def create_directory(self, job_id: str) -> None:
        try:
            self._job_dir(job_id).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise JobDirectoryCreationError(job_id) from e
        except Exception as exc:
            raise JobDirectoryCreationError(job_id) from exc

    @override
    def remove(self, job_id: str) -> bool:
        try:
            shutil.rmtree(self._job_dir(job_id), ignore_errors=False)
            return True
        except OSError:
            return False

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    @override
    async def save(
        self,
        job_id: str,
        relative_path: str,
        file: FileLike,
        *,
        mkdirs: bool = True,
    ) -> SavedJobFile:
        self.create_directory(job_id)

        dst = self._safe_path(job_id, relative_path)
        dst.parent.mkdir(parents=True, exist_ok=True)

        size = 0
        hasher = hashlib.sha256()

        # --------------------------------------------------------------
        # Case 1: bytes
        # --------------------------------------------------------------
        if isinstance(file, (bytes, bytearray)):
            async with aiofiles.open(dst, "wb") as f:
                _ = await f.write(file)
            size = len(file)
            hasher.update(file)

        # --------------------------------------------------------------
        # Case 2: filename / PathLike → copy
        # --------------------------------------------------------------
        elif isinstance(file, (str, PathLike)):
            src = Path(file).expanduser().resolve()
            if not src.is_file():
                raise FileNotFoundError(src)

            _ = shutil.copyfile(src, dst)
            size = dst.stat().st_size

            with open(dst, "rb") as f:
                for chunk in iter(lambda: f.read(self._CHUNK_SIZE), b""):
                    hasher.update(chunk)

        # --------------------------------------------------------------
        # Case 3: async file-like
        # --------------------------------------------------------------
        else:
            async with aiofiles.open(dst, "wb") as f:
                while True:
                    chunk = await file.read(self._CHUNK_SIZE)
                    if not chunk:
                        break
                    _ = await f.write(chunk)
                    size += len(chunk)
                    hasher.update(chunk)

        return SavedJobFile(
            relative_path=relative_path,
            size=size,
            hash=hasher.hexdigest(),
        )

    @override
    def allocate_path(
        self,
        job_id: str,
        relative_path: str,
        *,
        mkdirs: bool = True,
    ) -> Path:
        self.create_directory(job_id)
        path = self._safe_path(job_id, relative_path)
        if mkdirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Reading / resolving
    # ------------------------------------------------------------------

    @override
    async def open(
        self,
        job_id: str,
        relative_path: str,
    ) -> AsyncFileLike:
        path = self._safe_path(job_id, relative_path)
        return await aiofiles.open(path, "rb")

    @override
    def resolve_path(
        self,
        job_id: str,
        relative_path: str | None = None,
    ) -> Path:
        return self._safe_path(job_id, relative_path)
