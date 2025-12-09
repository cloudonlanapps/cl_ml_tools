"""Master module - dynamic route aggregator for FastAPI."""

from typing import Callable

from importlib.metadata import entry_points

from fastapi import APIRouter

from .common.job_repository import JobRepository
from .common.file_storage import FileStorage


def create_master_router(
    repository: JobRepository, file_storage: FileStorage, get_current_user: Callable
) -> APIRouter:
    """Dynamically aggregate all plugin routes from entry points.

    Discovers routes from [project.entry-points."cl_ml_tools.routes"]
    in pyproject.toml and creates a combined router.

    Args:
        repository: JobRepository implementation for job persistence
        file_storage: FileStorage implementation for file operations
        get_current_user: Callable dependency for authentication.
                          Should return user object or None.

    Returns:
        Combined APIRouter with all plugin routes

    Raises:
        RuntimeError: If a plugin fails to load (missing dependency, etc.)

    Example:
        from fastapi import FastAPI
        from cl_ml_tools.master import create_master_router

        app = FastAPI()

        # Your implementations
        repository = SQLiteJobRepository("./jobs.db")
        file_storage = LocalFileStorage("./media")

        async def get_current_user():
            return None  # Or return user from JWT

        # Mount all plugin routes
        app.include_router(
            create_master_router(repository, file_storage, get_current_user),
            prefix="/api"
        )
    """
    master = APIRouter()

    # Discover all route factories from entry points
    eps = entry_points(group="cl_ml_tools.routes")

    for ep in eps:
        try:
            create_router = ep.load()  # Load the factory function
            plugin_router = create_router(repository, file_storage, get_current_user)
            master.include_router(plugin_router, tags=[ep.name])
        except Exception as e:
            # Plugin dependency missing = exception (fail fast)
            raise RuntimeError(f"Failed to load plugin '{ep.name}': {e}")

    return master


def get_available_plugins() -> list:
    """Get list of available plugins.

    Returns:
        List of plugin names registered as entry points
    """
    eps = entry_points(group="cl_ml_tools.routes")
    return [ep.name for ep in eps]
