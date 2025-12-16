# Contributing to cl_ml_tools

Thank you for your interest in contributing! This guide will help you create high-quality plugins that follow the project's patterns and best practices.

## Table of Contents

- [Plugin Development Workflow](#plugin-development-workflow)
- [Plugin Architecture](#plugin-architecture)
- [Step-by-Step Plugin Creation](#step-by-step-plugin-creation)
- [Testing Requirements](#testing-requirements)
- [Code Style Guidelines](#code-style-guidelines)
- [Submitting Your Contribution](#submitting-your-contribution)

---

## Plugin Development Workflow

1. **Plan your plugin** - Define what task it performs, what inputs/outputs it needs
2. **Create plugin structure** - Follow the standard directory layout
3. **Implement the algorithm** - Pure Python logic in `algo/` directory
4. **Define schemas** - Pydantic models for parameters and outputs
5. **Implement the task** - ComputeModule that orchestrates the algorithm
6. **Create routes** - FastAPI endpoint factory for job creation
7. **Write comprehensive tests** - Schema, task execution, algorithm unit tests
8. **Document** - README.md with examples and parameter documentation
9. **Register** - Add entry points to `pyproject.toml`
10. **Submit PR** - Submit your contribution with clear description

---

## Plugin Architecture

### Standard Directory Structure

```
src/cl_ml_tools/plugins/your_plugin/
├── __init__.py          # Public exports (Task, Params, Result schemas)
├── schema.py            # Pydantic parameter and result models
├── task.py              # ComputeModule implementation
├── routes.py            # FastAPI route factory
├── algo/                # Pure computation functions
│   ├── __init__.py
│   ├── algorithm1.py   # Modular algorithm implementations
│   └── algorithm2.py
└── README.md            # Plugin documentation
```

### The `algo/` Pattern

**Purpose:** Separate pure computation logic from framework code (FastAPI, Pydantic, etc.)

**Benefits:**
- ✅ Framework-agnostic - can be reused in other contexts
- ✅ Easy to test - no async, no mocking frameworks
- ✅ Single responsibility - one function, one computation
- ✅ Maintainable - clear separation of concerns

**Guidelines:**
- Pure Python functions (no FastAPI, Pydantic, or async)
- Accept simple types (str, int, numpy arrays, PIL images)
- Return simple types or raise clear exceptions
- Include type hints
- Add logging for debugging (not for user communication)

**Example: `algo/image_processor.py`**
```python
"""Pure image processing algorithms."""
from pathlib import Path
import logging
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

def resize_image(image_path: str | Path, target_size: tuple[int, int]) -> Image.Image:
    """Resize image to target dimensions.

    Args:
        image_path: Path to input image
        target_size: (width, height) in pixels

    Returns:
        Resized PIL Image

    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If target_size is invalid
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if target_size[0] <= 0 or target_size[1] <= 0:
        raise ValueError(f"Invalid target size: {target_size}")

    logger.info(f"Resizing {image_path} to {target_size}")

    img = Image.open(image_path)
    resized = img.resize(target_size, Image.Resampling.LANCZOS)

    return resized
```

---

## Step-by-Step Plugin Creation

### Step 1: Create Plugin Directory

```bash
mkdir -p src/cl_ml_tools/plugins/your_plugin/algo
touch src/cl_ml_tools/plugins/your_plugin/__init__.py
touch src/cl_ml_tools/plugins/your_plugin/schema.py
touch src/cl_ml_tools/plugins/your_plugin/task.py
touch src/cl_ml_tools/plugins/your_plugin/routes.py
touch src/cl_ml_tools/plugins/your_plugin/algo/__init__.py
touch src/cl_ml_tools/plugins/your_plugin/README.md
```

### Step 2: Define Schemas (`schema.py`)

```python
"""Your plugin parameters and result schemas."""
from pydantic import BaseModel, Field
from ...common.schemas import BaseJobParams

class YourPluginParams(BaseJobParams):
    """Parameters for your plugin task.

    Attributes:
        input_paths: List of absolute paths to input files (inherited)
        output_paths: List of absolute paths for outputs (inherited)
        custom_param: Your custom parameter
    """
    custom_param: str = Field(
        default="default_value",
        description="Description of what this parameter does"
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold value between 0 and 1"
    )

class YourPluginResult(BaseModel):
    """Result for a single processed file."""
    file_path: str = Field(..., description="Path to the processed file")
    status: str = Field(..., description="'success' or 'error'")
    output: dict | None = Field(None, description="Processing results")
    error: str | None = Field(None, description="Error message if status is 'error'")
```

**Best Practices:**
- Extend `BaseJobParams` (provides `input_paths` and `output_paths`)
- Use `Field()` with descriptions for all parameters
- Add validation constraints (`ge`, `le`, `min_length`, etc.)
- Include comprehensive docstrings

### Step 3: Implement Algorithm (`algo/your_algorithm.py`)

```python
"""Core algorithm implementation."""
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def process_file(
    input_path: str | Path,
    output_path: str | Path,
    threshold: float
) -> dict:
    """Process a single file.

    Args:
        input_path: Path to input file
        output_path: Path to save output
        threshold: Processing threshold

    Returns:
        Dict with processing results

    Raises:
        FileNotFoundError: If input doesn't exist
        ValueError: If threshold is invalid
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    logger.info(f"Processing {input_path} with threshold={threshold}")

    # Your processing logic here
    result = {"processed": True, "threshold_used": threshold}

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # ... save logic ...

    return result
```

### Step 4: Implement Task (`task.py`)

```python
"""Your plugin task implementation."""
import logging
from typing import Callable, override

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from .algo.your_algorithm import process_file
from .schema import YourPluginParams, YourPluginResult

logger = logging.getLogger(__name__)

class YourPluginTask(ComputeModule[YourPluginParams]):
    """Compute module for your plugin."""

    @property
    @override
    def task_type(self) -> str:
        return "your_plugin"

    @override
    def get_schema(self) -> type[BaseJobParams]:
        return YourPluginParams

    @override
    async def execute(
        self,
        job: Job,
        params: YourPluginParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        """Execute your plugin task.

        Args:
            job: Job instance
            params: Validated parameters
            progress_callback: Optional callback for progress (0-100)

        Returns:
            TaskResult with status and outputs
        """
        try:
            file_results: list[dict] = []
            total_files = len(params.input_paths)

            for index, (input_path, output_path) in enumerate(
                zip(params.input_paths, params.output_paths)
            ):
                try:
                    # Call your algorithm
                    result = process_file(
                        input_path=input_path,
                        output_path=output_path,
                        threshold=params.threshold
                    )

                    # Create result object
                    file_result = YourPluginResult(
                        file_path=input_path,
                        status="success",
                        output=result
                    )
                    file_results.append(file_result.model_dump())

                except FileNotFoundError:
                    logger.error(f"File not found: {input_path}")
                    file_results.append(
                        YourPluginResult(
                            file_path=input_path,
                            status="error",
                            error="File not found"
                        ).model_dump()
                    )

                except Exception as e:
                    logger.error(f"Failed to process {input_path}: {e}")
                    file_results.append(
                        YourPluginResult(
                            file_path=input_path,
                            status="error",
                            error=str(e)
                        ).model_dump()
                    )

                # Report progress
                if progress_callback:
                    progress = int((index + 1) / total_files * 100)
                    progress_callback(progress)

            # Determine overall status
            all_success = all(r["status"] == "success" for r in file_results)

            return {
                "status": "ok" if all_success else "error",
                "task_output": {
                    "files": file_results,
                    "total_files": total_files,
                },
            }

        except Exception as e:
            logger.exception(f"Unexpected error in YourPluginTask: {e}")
            return {"status": "error", "error": f"Task failed: {str(e)}"}
```

**Best Practices:**
- Handle errors gracefully (per-file and overall)
- Always call `progress_callback` if provided
- Use `logger` for debugging, not `print()`
- Return structured `TaskResult` dict

### Step 5: Create Routes (`routes.py`)

```python
"""FastAPI route factory for your plugin."""
from typing import Annotated, Callable, Literal, Protocol, TypedDict
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ...common.file_storage import FileStorage
from ...common.job_repository import JobRepository
from ...common.schemas import Job

class UserLike(Protocol):
    id: str | None

class JobCreatedResponse(TypedDict):
    job_id: str
    status: Literal["queued"]

def create_router(
    repository: JobRepository,
    file_storage: FileStorage,
    get_current_user: Callable[[], UserLike | None],
) -> APIRouter:
    """Create FastAPI router for your plugin endpoints.

    Args:
        repository: Job repository for persistence
        file_storage: File storage for managing uploaded files
        get_current_user: Dependency for getting current user

    Returns:
        Configured APIRouter with your plugin endpoint
    """
    router = APIRouter()

    @router.post("/jobs/your_plugin", response_model=JobCreatedResponse)
    async def create_your_plugin_job(
        file: Annotated[UploadFile, File(description="Input file")],
        threshold: Annotated[float, Form(ge=0.0, le=1.0, description="Processing threshold")] = 0.5,
        priority: Annotated[int, Form(ge=0, le=10, description="Job priority (0-10)")] = 5,
        user: Annotated[UserLike | None, Depends(get_current_user)] = None,
    ) -> JobCreatedResponse:
        """Create a new your_plugin job.

        Args:
            file: Uploaded input file
            threshold: Processing threshold
            priority: Job priority (0-10, higher is more priority)
            user: Current user (injected by dependency)

        Returns:
            JobCreatedResponse with job_id and status
        """
        job_id = str(uuid4())

        # Create job directory
        _ = file_storage.create_job_directory(job_id)

        if not file.filename:
            raise ValueError("Uploaded file has no filename")

        filename: str = file.filename

        # Save uploaded file
        file_info = await file_storage.save_input_file(job_id, filename, file)
        input_path = file_info["path"]

        # Define output path
        output_path = str(
            file_storage.get_output_path(job_id) / f"processed_{filename}"
        )

        # Create job
        job = Job(
            job_id=job_id,
            task_type="your_plugin",
            params={
                "input_paths": [input_path],
                "output_paths": [output_path],
                "threshold": threshold,
            },
        )

        created_by = user.id if user is not None else None
        _ = repository.add_job(job, created_by=created_by, priority=priority)

        return {
            "job_id": job_id,
            "status": "queued",
        }

    # Mark function as used (accessed via FastAPI decorator)
    _ = create_your_plugin_job

    return router
```

### Step 6: Update `__init__.py`

```python
"""Your plugin public exports."""
from .schema import YourPluginParams, YourPluginResult
from .task import YourPluginTask

__all__ = ["YourPluginTask", "YourPluginParams", "YourPluginResult"]
```

### Step 7: Register in `pyproject.toml`

```toml
[project.entry-points."cl_ml_tools.tasks"]
your_plugin = "cl_ml_tools.plugins.your_plugin.task:YourPluginTask"

[project.entry-points."cl_ml_tools.routes"]
your_plugin = "cl_ml_tools.plugins.your_plugin.routes:create_router"
```

---

## Testing Requirements

### Test File Structure

Create `tests/test_your_plugin.py` with three test classes:

```python
"""Comprehensive test suite for your plugin."""
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from cl_ml_tools.common.schemas import Job
from cl_ml_tools.plugins.your_plugin.schema import YourPluginParams, YourPluginResult
from cl_ml_tools.plugins.your_plugin.task import YourPluginTask

# ============================================================================
# Test Class 1: Schema Validation
# ============================================================================

class TestYourPluginParams:
    """Test YourPluginParams schema validation."""

    def test_default_params(self) -> None:
        """Test default parameter values."""
        params = YourPluginParams(input_paths=["/test/file.jpg"], output_paths=["/out/file.jpg"])

        assert params.input_paths == ["/test/file.jpg"]
        assert params.threshold == 0.5  # default

    def test_invalid_threshold(self) -> None:
        """Test that invalid threshold is rejected."""
        with pytest.raises(Exception):  # Pydantic validation error
            YourPluginParams(
                input_paths=["/test/file.jpg"],
                output_paths=["/out/file.jpg"],
                threshold=1.5  # Invalid (>1.0)
            )

# ============================================================================
# Test Class 2: Task Execution
# ============================================================================

class TestYourPluginTask:
    """Test YourPluginTask execution."""

    @pytest.mark.asyncio
    async def test_execute_with_mocked_algorithm(self, tmp_path: Path) -> None:
        """Test task execution with mocked algorithm."""
        # Create test file
        test_file = tmp_path / "input.txt"
        test_file.write_text("test content")

        output_file = tmp_path / "output.txt"

        # Create job
        job = Job(
            job_id="test-job-001",
            task_type="your_plugin",
            params={
                "input_paths": [str(test_file)],
                "output_paths": [str(output_file)],
                "threshold": 0.7,
            }
        )

        params = YourPluginParams(**job.params)
        task = YourPluginTask()

        # Execute
        result = await task.execute(job, params, None)

        # Verify
        assert result["status"] == "ok"
        assert result["task_output"]["total_files"] == 1

# ============================================================================
# Test Class 3: Algorithm Unit Tests
# ============================================================================

class TestYourPluginAlgorithm:
    """Test your plugin algorithm functions."""

    def test_process_file_success(self, tmp_path: Path) -> None:
        """Test successful file processing."""
        from cl_ml_tools.plugins.your_plugin.algo.your_algorithm import process_file

        input_file = tmp_path / "input.txt"
        input_file.write_text("test")

        output_file = tmp_path / "output.txt"

        result = process_file(str(input_file), str(output_file), threshold=0.5)

        assert result["processed"] is True
        assert output_file.exists()
```

### Test Coverage Requirements

- **Minimum:** 80% code coverage for your plugin
- **Schema tests:** All parameters, validation, edge cases
- **Task tests:** Success, errors, progress callbacks, multiple files
- **Algorithm tests:** Core logic, error handling, edge cases

### Running Tests

```bash
# Run your plugin tests only
uv run pytest tests/test_your_plugin.py -v

# Run with coverage
uv run pytest tests/test_your_plugin.py --cov=src/cl_ml_tools/plugins/your_plugin --cov-report=html

# Run all tests
uv run pytest
```

---

## Code Style Guidelines

### Linting and Type Checking

```bash
# Run ruff linter
uv run ruff check src/cl_ml_tools/plugins/your_plugin/

# Fix auto-fixable issues
uv run ruff check --fix src/cl_ml_tools/plugins/your_plugin/

# Run type checker
uv run basedpyright src/cl_ml_tools/plugins/your_plugin/
```

### Code Style Rules

1. **Type hints:** Always include type hints for function parameters and returns
2. **Docstrings:** Use Google-style docstrings for all public functions/classes
3. **Imports:** Use absolute imports, sorted by `ruff`
4. **Line length:** Max 100 characters (configured in `pyproject.toml`)
5. **Naming:**
   - Classes: `PascalCase`
   - Functions/variables: `snake_case`
   - Constants: `UPPER_SNAKE_CASE`
6. **Logging:** Use `logger` from `logging`, not `print()`
7. **Error handling:** Raise specific exceptions with clear messages

---

## Submitting Your Contribution

### Before Submitting

- [ ] All tests pass (`uv run pytest`)
- [ ] Code coverage >80% for your plugin
- [ ] Ruff linter passes (`uv run ruff check`)
- [ ] Type checker passes (`uv run basedpyright`)
- [ ] README.md created with usage examples
- [ ] Entry points registered in `pyproject.toml`

### Pull Request Checklist

1. **Title:** Clear, descriptive title (`Add: Your Plugin Name`)
2. **Description:**
   - What the plugin does
   - Use cases
   - Any special requirements (dependencies, models, etc.)
3. **Changes:**
   - List files added/modified
   - Note any breaking changes
4. **Testing:**
   - Describe how you tested
   - Include test coverage percentage
5. **Documentation:**
   - Link to plugin README
   - Note any updates to main README.md

### Example PR Description

```markdown
## Add: Watermark Plugin

### Description
Adds a watermark plugin for adding text/image watermarks to photos.

### Use Cases
- Copyright protection
- Branding
- Batch watermarking

### Changes
- Created `src/cl_ml_tools/plugins/watermark/` with full implementation
- Added 15 comprehensive tests (100% coverage)
- Registered entry points in `pyproject.toml`
- Created plugin README with examples

### Dependencies
- Pillow (already in dependencies)

### Testing
- All 15 tests pass
- Tested with JPEG, PNG, WebP formats
- Verified watermark positioning and opacity

### Documentation
- Plugin README: `src/cl_ml_tools/plugins/watermark/README.md`
- Updated main README with watermark plugin info
```

---

## Questions or Issues?

- **Bug reports:** [GitHub Issues](https://github.com/your-repo/cl_ml_tools/issues)
- **Feature requests:** [GitHub Discussions](https://github.com/your-repo/cl_ml_tools/discussions)
- **Questions:** Check existing issues or start a discussion

Thank you for contributing to cl_ml_tools! 🎉
