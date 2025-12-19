"""Face embedding task implementation."""

import logging
from typing import Callable, override

import numpy as np

from ...common.compute_module import ComputeModule
from ...common.job_storage import JobStorage
from .algo.face_embedder import FaceEmbedder
from .schema import FaceEmbeddingOutput, FaceEmbeddingParams

logger = logging.getLogger(__name__)


class FaceEmbeddingTask(ComputeModule[FaceEmbeddingParams, FaceEmbeddingOutput]):
    """Compute module for generating a face embedding using ONNX model."""

    schema: type[FaceEmbeddingParams] = FaceEmbeddingParams

    def __init__(self) -> None:
        self._embedder: FaceEmbedder | None = None

    @property
    @override
    def task_type(self) -> str:
        return "face_embedding"

    @override
    def setup(self) -> None:
        if self._embedder is None:
            try:
                self._embedder = FaceEmbedder()
                logger.info("Face embedder initialized successfully")
            except (FileNotFoundError, RuntimeError, ImportError, OSError) as exc:
                logger.error("Face embedder initialization failed", exc_info=exc)
                raise RuntimeError(
                    "Failed to initialize face embedder. "
                    + "Ensure ONNX Runtime is installed and the model is available."
                ) from exc

    @override
    async def run(
        self,
        job_id: str,
        params: FaceEmbeddingParams,
        storage: JobStorage,
        progress_callback: Callable[[int], None] | None = None,
    ) -> FaceEmbeddingOutput:
        if not self._embedder:
            raise RuntimeError("Face embedder is not initialized")

        input_path = storage.resolve_path(job_id, params.input_path)

        embedding_array, quality_score = self._embedder.embed(
            image_path=str(input_path),
            normalize=params.normalize,
            compute_quality=True,
        )

        path = storage.allocate_path(
            job_id=job_id,
            relative_path=params.output_path,
        )
        np.save(path, embedding_array)

        if progress_callback:
            progress_callback(100)

        return FaceEmbeddingOutput(
            normalized=params.normalize,
            embedding_dim=int(embedding_array.shape[0]),
            quality_score=quality_score,
        )
