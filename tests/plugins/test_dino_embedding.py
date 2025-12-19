"""Unit and integration tests for DINO embedding plugin.

Tests schema validation, 384-dim embedding generation, normalization, task execution, routes, and full job lifecycle.
Requires ML models downloaded.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from cl_ml_tools import Worker
    from cl_ml_tools.common.file_storage import JobStorage
    from cl_ml_tools.common.job_repository import JobRepository

from cl_ml_tools.plugins.dino_embedding.schema import (
    DinoEmbeddingOutput,
    DinoEmbeddingParams,
)
from cl_ml_tools.plugins.dino_embedding.task import DinoEmbeddingTask

# ============================================================================
# SCHEMA TESTS
# ============================================================================


def test_dino_embedding_params_schema_validation():
    """Test DinoEmbeddingParams schema validates correctly."""
    params = DinoEmbeddingParams(
        input_path="/path/to/input.jpg",
        output_path="output/embedding.npy",
        normalize=True,
    )

    assert params.input_path == "/path/to/input.jpg"
    assert params.output_path == "output/embedding.npy"
    assert params.normalize is True


def test_dino_embedding_params_defaults():
    """Test DinoEmbeddingParams has correct default values."""
    params = DinoEmbeddingParams(
        input_path="/path/to/input.jpg",
        output_path="output/embedding.npy",
    )

    assert params.normalize is True


def test_dino_embedding_output_schema_validation():
    """Test DinoEmbeddingOutput schema validates correctly."""
    output = DinoEmbeddingOutput(
        normalized=True,
        embedding_dim=384,
    )

    assert output.normalized is True
    assert output.embedding_dim == 384


# ============================================================================
# ALGORITHM TESTS
# ============================================================================


@pytest.mark.requires_models
def test_dino_algo_basic(sample_image_path: Path):
    """Test basic DINO embedding generation."""
    from cl_ml_tools.plugins.dino_embedding.algo.dino_embedder import DinoEmbedder

    embedder = DinoEmbedder()
    embedding = embedder.embed(str(sample_image_path), normalize=True)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)


@pytest.mark.requires_models
def test_dino_algo_384_dim_output(sample_image_path: Path):
    """Test DINO outputs 384-dimensional embeddings."""
    from cl_ml_tools.plugins.dino_embedding.algo.dino_embedder import DinoEmbedder

    embedder = DinoEmbedder()
    embedding = embedder.embed(str(sample_image_path), normalize=False)

    assert embedding.shape[0] == 384


@pytest.mark.requires_models
def test_dino_algo_normalization(sample_image_path: Path):
    """Test DINO embedding normalization."""
    from cl_ml_tools.plugins.dino_embedding.algo.dino_embedder import DinoEmbedder

    embedder = DinoEmbedder()
    embedding_norm = embedder.embed(str(sample_image_path), normalize=True)
    norm = np.linalg.norm(embedding_norm)

    assert abs(norm - 1.0) < 0.01


@pytest.mark.requires_models
def test_dino_algo_consistency(sample_image_path: Path):
    """Test DINO embeddings are consistent."""
    from cl_ml_tools.plugins.dino_embedding.algo.dino_embedder import DinoEmbedder

    embedder = DinoEmbedder()

    embedding1 = embedder.embed(str(sample_image_path), normalize=True)
    embedding2 = embedder.embed(str(sample_image_path), normalize=True)

    assert np.allclose(embedding1, embedding2, rtol=1e-5)


@pytest.mark.requires_models
def test_dino_algo_different_images(sample_image_path: Path, synthetic_image: Path):
    """Test different images produce different DINO embeddings."""
    from cl_ml_tools.plugins.dino_embedding.algo.dino_embedder import DinoEmbedder

    embedder = DinoEmbedder()

    embedding1 = embedder.embed(str(sample_image_path), normalize=True)
    embedding2 = embedder.embed(str(synthetic_image), normalize=True)

    assert not np.allclose(embedding1, embedding2, rtol=0.1)


# ============================================================================
# TASK TESTS
# ============================================================================


@pytest.mark.requires_models
@pytest.mark.asyncio
async def test_dino_task_run_success(sample_image_path: Path, tmp_path: Path):
    """Test DinoEmbeddingTask execution success."""
    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(sample_image_path.read_bytes())

    params = DinoEmbeddingParams(
        input_path=str(input_path),
        output_path="output/embedding.npy",
        normalize=True,
    )

    task = DinoEmbeddingTask()
    task.setup()
    job_id = "test-job-123"

    class MockStorage:
        def create_directory(self, job_id: str) -> None:
            pass

        def remove(self, job_id: str) -> bool:
            return True

        async def save(
            self, job_id: str, relative_path: str, file: Any, *, mkdirs: bool = True
        ) -> Any:
            return None

        async def open(self, job_id: str, relative_path: str) -> Any:
            return None

        def resolve_path(self, job_id: str, relative_path: str | None = None) -> Path:
            return tmp_path / job_id / (relative_path or "")

        def allocate_path(self, job_id: str, relative_path: str, *, mkdirs: bool = True) -> Path:
            output_path = tmp_path / "output" / "embedding.npy"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return output_path

    storage = MockStorage()

    output = await task.run(job_id, params, storage)

    assert isinstance(output, DinoEmbeddingOutput)
    assert output.embedding_dim == 384
    assert output.normalized is True


@pytest.mark.requires_models
@pytest.mark.asyncio
async def test_dino_task_run_file_not_found(tmp_path: Path):
    """Test DinoEmbeddingTask raises FileNotFoundError for missing input."""
    params = DinoEmbeddingParams(
        input_path="/nonexistent/file.jpg",
        output_path="output/embedding.npy",
    )

    task = DinoEmbeddingTask()
    task.setup()
    job_id = "test-job-789"

    class MockStorage:
        def create_directory(self, job_id: str) -> None:
            pass

        def remove(self, job_id: str) -> bool:
            return True

        async def save(
            self, job_id: str, relative_path: str, file: Any, *, mkdirs: bool = True
        ) -> Any:
            return None

        async def open(self, job_id: str, relative_path: str) -> Any:
            return None

        def resolve_path(self, job_id: str, relative_path: str | None = None) -> Path:
            return tmp_path / job_id / (relative_path or "")

        def allocate_path(self, job_id: str, relative_path: str, *, mkdirs: bool = True) -> Path:
            return tmp_path / "output" / "embedding.npy"

    storage = MockStorage()

    with pytest.raises(FileNotFoundError):
        await task.run(job_id, params, storage)


# ============================================================================
# ROUTE TESTS
# ============================================================================


def test_dino_embedding_route_creation(api_client: "TestClient"):
    """Test dino_embedding route is registered."""
    response = api_client.get("/openapi.json")
    assert response.status_code == 200

    openapi = response.json()
    assert "/jobs/dino_embedding" in openapi["paths"]


@pytest.mark.requires_models
def test_dino_embedding_route_job_submission(api_client: "TestClient", sample_image_path: Path):
    """Test job submission via dino_embedding route."""
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/dino_embedding",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"normalize": "true", "priority": "5"},
        )

    assert response.status_code == 200
    data = response.json()

    assert "job_id" in data
    assert data["status"] == "queued"
    assert data["task_type"] == "dino_embedding"


# ============================================================================
# INTEGRATION TEST (API → Worker)
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_models
async def test_dino_embedding_full_job_lifecycle(
    api_client: "TestClient",
    worker: "Worker",
    job_repository: "JobRepository",
    sample_image_path: Path,
    file_storage: "JobStorage",
):
    """Test complete flow: API → Repository → Worker → Output."""
    # 1. Submit job via API
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/dino_embedding",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"normalize": "true", "priority": "5"},
        )

    assert response.status_code == 200
    job_id = response.json()["job_id"]

    # 2. Worker consumes job
    processed = await worker.run_once()
    assert processed == 1

    # 3. Verify completion
    job = job_repository.get_job(job_id)
    assert job is not None
    assert job.status == "completed"

    # 4. Validate output
    assert job.output is not None
    output = DinoEmbeddingOutput.model_validate(job.output)
    assert output.embedding_dim == 384
    assert output.normalized is True
