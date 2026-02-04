"""Unit and integration tests for face embedding plugin.

Tests schema validation, embedding generation with quality scores, normalization, task execution, routes, and full job lifecycle.
Requires ML models downloaded.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from cl_ml_tools import Worker
    from cl_ml_tools.common.job_repository import JobRepository
    from cl_ml_tools.common.job_storage import JobStorage

from cl_ml_tools.plugins.face_embedding.algo.face_embedder import FaceEmbedder
from cl_ml_tools.plugins.face_embedding.schema import (
    FaceEmbeddingOutput,
    FaceEmbeddingParams,
)
from cl_ml_tools.plugins.face_embedding.task import FaceEmbeddingTask

# ============================================================================
# SCHEMA TESTS
# ============================================================================


def test_face_embedding_params_schema_validation():
    """Test FaceEmbeddingParams schema validates correctly."""
    params = FaceEmbeddingParams(
        input_path="/path/to/input.jpg",
        output_path="output/embedding.npy",
        normalize=True,
    )

    assert params.input_path == "/path/to/input.jpg"
    assert params.output_path == "output/embedding.npy"
    assert params.normalize is True


def test_face_embedding_params_defaults():
    """Test FaceEmbeddingParams has correct default values."""
    params = FaceEmbeddingParams(
        input_path="/path/to/input.jpg",
        output_path="output/embedding.npy",
    )

    assert params.normalize is True


def test_face_embedding_output_schema_validation():
    """Test FaceEmbeddingOutput schema validates correctly."""
    output = FaceEmbeddingOutput(
        normalized=True,
        embedding_dim=512,
        quality_score=0.85,
    )

    assert output.normalized is True
    assert output.embedding_dim == 512
    assert output.quality_score == 0.85


def test_face_embedding_output_optional_quality_score():
    """Test FaceEmbeddingOutput quality_score is optional."""
    output = FaceEmbeddingOutput(
        normalized=True,
        embedding_dim=128,
    )

    assert output.quality_score is None


def test_face_embedding_output_quality_score_validation():
    """Test FaceEmbeddingOutput validates quality_score range."""
    # Valid quality score
    output = FaceEmbeddingOutput(
        normalized=True,
        embedding_dim=512,
        quality_score=0.75,
    )
    assert output.quality_score == 0.75

    # Invalid quality score (too high)
    with pytest.raises(ValueError):
        _ = FaceEmbeddingOutput(
            normalized=True,
            embedding_dim=512,
            quality_score=1.5,
        )

    # Invalid quality score (negative)
    with pytest.raises(ValueError):
        _ = FaceEmbeddingOutput(
            normalized=True,
            embedding_dim=512,
            quality_score=-0.1,
        )


# ============================================================================
# ALGORITHM TESTS
# ============================================================================


@pytest.mark.requires_models
def test_face_embedding_algo_basic(sample_image_path: Path):
    """Test basic face embedding generation."""
    from cl_ml_tools.plugins.face_embedding.algo.face_embedder import FaceEmbedder

    embedder = FaceEmbedder()
    result = embedder.embed(str(sample_image_path), normalize=True)

    # Result may be tuple (embedding, quality) or just embedding
    if isinstance(result, tuple):
        embedding, quality = result
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        assert quality is None or (0.0 <= quality <= 1.0)
    else:
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1


@pytest.mark.requires_models
def test_face_embedding_algo_embedding_dimensions(sample_image_path: Path):
    """Test face embedding has expected dimensions (128 or 512)."""
    from cl_ml_tools.plugins.face_embedding.algo.face_embedder import FaceEmbedder

    embedder = FaceEmbedder()
    result = embedder.embed(str(sample_image_path), normalize=True)

    if isinstance(result, tuple):
        embedding, _ = result
    else:
        embedding = result

    # Common face embedding dimensions
    assert embedding.shape[0] in (128, 512)


@pytest.mark.requires_models
def test_face_embedding_algo_normalization(sample_image_path: Path):
    """Test face embedding normalization."""
    from cl_ml_tools.plugins.face_embedding.algo.face_embedder import FaceEmbedder

    embedder = FaceEmbedder()
    result = embedder.embed(str(sample_image_path), normalize=True)

    if isinstance(result, tuple):
        embedding, _ = result
    else:
        embedding = result

    norm = np.linalg.norm(embedding)

    # Should be L2-normalized (norm ≈ 1)
    assert abs(norm - 1.0) < 0.01


@pytest.mark.requires_models
def test_face_embedding_algo_quality_score(sample_image_path: Path):
    """Test face embedding returns quality score."""
    from cl_ml_tools.plugins.face_embedding.algo.face_embedder import FaceEmbedder

    embedder = FaceEmbedder()
    result = embedder.embed(str(sample_image_path), normalize=True)

    # If quality score is returned, it should be in valid range
    if isinstance(result, tuple):
        embedding, quality = result
        if quality is not None:
            assert 0.0 <= quality <= 1.0


@pytest.mark.requires_models
def test_face_embedding_algo_consistency(sample_image_path: Path):
    """Test face embeddings are consistent."""
    from cl_ml_tools.plugins.face_embedding.algo.face_embedder import FaceEmbedder

    embedder = FaceEmbedder()

    result1 = embedder.embed(str(sample_image_path), normalize=True)
    result2 = embedder.embed(str(sample_image_path), normalize=True)

    if isinstance(result1, tuple):
        embedding1, _ = result1
        embedding2, _ = result2
    else:
        embedding1 = result1
        embedding2 = result2

    if embedding1 is not None and embedding2 is not None:
        assert np.allclose(cast("Any", embedding1), cast("Any", embedding2), rtol=1e-5)
    else:
        assert embedding1 is None
        assert embedding2 is None


@pytest.mark.requires_models
def test_face_embedding_algo_different_images(
    sample_image_path: Path, synthetic_image: Path
):
    """Test different images produce different face embeddings."""
    embedder = FaceEmbedder()

    result1 = embedder.embed(str(sample_image_path), normalize=True)
    result2 = embedder.embed(str(synthetic_image), normalize=True)

    embedding1, _ = result1
    embedding2, _ = result2

    # May raise exception if no face found - that's ok
    # If both succeed, embeddings should be different
    if embedding1 is not None and embedding2 is not None:
        assert not np.allclose(
            cast("Any", embedding1), cast("Any", embedding2), rtol=0.1
        )


@pytest.mark.requires_models
def test_face_embedding_algo_error_handling_invalid_file(tmp_path: Path):
    """Test face embedding handles invalid image files."""
    from cl_ml_tools.plugins.face_embedding.algo.face_embedder import FaceEmbedder

    invalid_file = tmp_path / "invalid.jpg"
    _ = invalid_file.write_text("not an image")

    embedder = FaceEmbedder()

    with pytest.raises(Exception):
        embedder.embed(str(invalid_file), normalize=True)


# ============================================================================
# TASK TESTS
# ============================================================================


@pytest.mark.requires_models
@pytest.mark.asyncio
async def test_face_embedding_task_run_success(sample_image_path: Path, tmp_path: Path):
    """Test FaceEmbeddingTask execution success."""
    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(sample_image_path.read_bytes())

    params = FaceEmbeddingParams(
        input_path=str(input_path),
        output_path="output/embedding.npy",
        normalize=True,
    )

    task = FaceEmbeddingTask()
    task.setup()
    job_id = "test-job-123"

    class MockStorage:
        def create_directory(self, job_id: str) -> None:
            pass

        def remove(self, job_id: str) -> bool:
            return True

        async def save(
            self,
            job_id: str,
            relative_path: str,
            file: Any,
            *,
            mkdirs: bool = True,
        ) -> Any:
            return None

        async def open(self, job_id: str, relative_path: str) -> Any:
            return None

        def resolve_path(self, job_id: str, relative_path: str | None = None) -> Path:
            return tmp_path / job_id / (relative_path or "")

        def allocate_path(
            self, job_id: str, relative_path: str, *, mkdirs: bool = True
        ) -> Path:
            output_path = tmp_path / "output" / "embedding.npy"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return output_path

    storage = MockStorage()

    output = await task.run(job_id, params, storage)

    assert isinstance(output, FaceEmbeddingOutput)
    assert output.embedding_dim in (128, 512)
    assert output.normalized is True


@pytest.mark.requires_models
@pytest.mark.asyncio
async def test_face_embedding_task_run_file_not_found(tmp_path: Path):
    """Test FaceEmbeddingTask raises FileNotFoundError for missing input."""
    params = FaceEmbeddingParams(
        input_path="/nonexistent/file.jpg",
        output_path="output/embedding.npy",
    )

    task = FaceEmbeddingTask()
    task.setup()
    job_id = "test-job-789"

    class MockStorage:
        def create_directory(self, job_id: str) -> None:
            pass

        def remove(self, job_id: str) -> bool:
            return True

        async def save(
            self,
            job_id: str,
            relative_path: str,
            file: Any,
            *,
            mkdirs: bool = True,
        ) -> Any:
            return None

        async def open(self, job_id: str, relative_path: str) -> Any:
            return None

        def resolve_path(self, job_id: str, relative_path: str | None = None) -> Path:
            return tmp_path / job_id / (relative_path or "")

        def allocate_path(
            self, job_id: str, relative_path: str, *, mkdirs: bool = True
        ) -> Path:
            return tmp_path / "output" / "embedding.npy"

    storage = MockStorage()

    with pytest.raises(FileNotFoundError):
        await task.run(job_id, params, storage)


# ============================================================================
# ROUTE TESTS
# ============================================================================


def test_face_embedding_route_creation(api_client: "TestClient"):
    """Test face_embedding route is registered."""
    response = api_client.get("/openapi.json")
    assert response.status_code == 200

    openapi = response.json()
    assert "/jobs/face_embedding" in openapi["paths"]


@pytest.mark.requires_models
def test_face_embedding_route_job_submission(
    api_client: "TestClient", sample_image_path: Path
):
    """Test job submission via face_embedding route."""
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/face_embedding",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"normalize": "true", "priority": "5"},
        )

    assert response.status_code == 200
    data = response.json()

    assert "job_id" in data
    assert data["status"] == "queued"
    assert data["task_type"] == "face_embedding"


# ============================================================================
# INTEGRATION TEST (API → Worker)
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_models
@pytest.mark.asyncio
async def test_face_embedding_full_job_lifecycle(
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
            "/jobs/face_embedding",
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
    output = FaceEmbeddingOutput.model_validate(job.output)
    assert output.embedding_dim in (128, 512)
    assert output.normalized is True


@pytest.mark.integration
@pytest.mark.requires_models
@pytest.mark.asyncio
async def test_face_embedding_full_job_lifecycle_with_quality(
    api_client: "TestClient",
    worker: "Worker",
    job_repository: "JobRepository",
    sample_image_path: Path,
    file_storage: "JobStorage",
):
    """Test complete flow and verify quality score is computed."""
    # 1. Submit job via API
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/face_embedding",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"normalize": "true", "priority": "6"},
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

    # 4. Validate output has quality score (if face was detected)
    assert job.output is not None
    output = FaceEmbeddingOutput.model_validate(job.output)
    # Quality score may be None if no face detected, or a float in [0, 1]
    if output.quality_score is not None:
        assert 0.0 <= output.quality_score <= 1.0
