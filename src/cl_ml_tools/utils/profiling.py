"""Performance profiling utilities for cl_ml_tools algorithms."""

import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from loguru import logger

P = ParamSpec("P")
R = TypeVar("R")


def timed(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to measure and log execution time of algorithm functions.

    Logs the function name and execution time at INFO level.

    Usage:
        @timed
        def my_algorithm(image):
            # ... processing ...
            return result
    """
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed_time = time.perf_counter() - start_time
            logger.info(
                f"[PROFILE] {func.__qualname__} took {elapsed_time:.3f}s"
            )

    return wrapper
