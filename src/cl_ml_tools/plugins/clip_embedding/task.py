"""MobileCLIP embedding task implementation."""

from typing import Callable, override

import numpy as np
from loguru import logger

from ...common.compute_module import ComputeModule
from ...common.job_storage import JobStorage
from .algo.clip_embedder import ClipEmbedder
from .schema import ClipEmbeddingOutput, ClipEmbeddingParams


class ClipEmbeddingTask(ComputeModule[ClipEmbeddingParams, ClipEmbeddingOutput]):
    """Compute module for generating MobileCLIP embeddings using ONNX model."""

    schema: type[ClipEmbeddingParams] = ClipEmbeddingParams

    def __init__(self) -> None:
        self._embedder: ClipEmbedder | None = None

    @property
    @override
    def task_type(self) -> str:
        return "clip_embedding"

    @override
    def setup(self) -> None:
        if self._embedder is None:
            try:
                self._embedder = ClipEmbedder()
                logger.info("MobileCLIP embedder initialized successfully")
            except (FileNotFoundError, RuntimeError, ImportError, OSError) as exc:
                logger.error("Failed to initialize MobileCLIP embedder", exc_info=exc)
                raise RuntimeError(
                    "Failed to initialize MobileCLIP embedder. "
                    + "Ensure ONNX Runtime is installed and the model is available."
                ) from exc

    @override
    async def run(
        self,
        job_id: str,
        params: ClipEmbeddingParams,
        storage: JobStorage,
        progress_callback: Callable[[int], None] | None = None,
    ) -> ClipEmbeddingOutput:
        if not self._embedder:
            raise RuntimeError("MobileCLIP embedder is not initialized")

        input_path = storage.resolve_path(job_id, params.input_path)

        embedding = self._embedder.embed(
            image_path=input_path,
            normalize=params.normalize,
        )

        path = storage.allocate_path(
            job_id=job_id,
            relative_path=params.output_path,
        )
        np.save(path, embedding)

        if progress_callback:
            progress_callback(100)

        return ClipEmbeddingOutput(
            embedding_dim=int(embedding.shape[0]),
            normalized=params.normalize,
        )
