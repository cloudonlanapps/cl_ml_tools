"""Performance profiling utilities for cl_ml_tools algorithms."""

import inspect
import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import ParamSpec, TypeVar, cast

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

        def log_elapsed():
            elapsed_time = time.perf_counter() - start_time
            logger.info(f"[PROFILE] {func.__qualname__} took {elapsed_time:.3f}s")

        if inspect.iscoroutinefunction(func):

            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                try:
                    result: R = await cast(Callable[P, Awaitable[R]], func)(*args, **kwargs)
                    return result
                finally:
                    log_elapsed()

            return cast(R, async_wrapper(*args, **kwargs))

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            log_elapsed()

    return wrapper
