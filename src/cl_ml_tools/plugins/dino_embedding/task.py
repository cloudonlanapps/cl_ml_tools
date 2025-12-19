"""DINOv2 embedding task implementation."""

import logging
from typing import Callable, override

import numpy as np

from ...common.compute_module import ComputeModule
from ...common.job_storage import JobStorage
from .algo.dino_embedder import DinoEmbedder
from .schema import DinoEmbeddingOutput, DinoEmbeddingParams

logger = logging.getLogger(__name__)


class DinoEmbeddingTask(ComputeModule[DinoEmbeddingParams, DinoEmbeddingOutput]):
    """Compute module for generating DINOv2 embeddings using ONNX model."""

    schema: type[DinoEmbeddingParams] = DinoEmbeddingParams

    def __init__(self) -> None:
        self._embedder: DinoEmbedder | None = None

    @property
    @override
    def task_type(self) -> str:
        return "dino_embedding"

    @override
    def setup(self) -> None:
        if self._embedder is None:
            try:
                self._embedder = DinoEmbedder()
                logger.info("DINOv2 embedder initialized successfully")
            except (FileNotFoundError, RuntimeError, ImportError, OSError) as exc:
                logger.error("Failed to initialize DINOv2 embedder", exc_info=exc)
                raise RuntimeError(
                    "Failed to initialize DINOv2 embedder. "
                    + "Ensure ONNX Runtime is installed and the model is available."
                ) from exc

    @override
    async def run(
        self,
        job_id: str,
        params: DinoEmbeddingParams,
        storage: JobStorage,
        progress_callback: Callable[[int], None] | None = None,
    ) -> DinoEmbeddingOutput:
        if not self._embedder:
            raise RuntimeError("DINOv2 embedder is not initialized")

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

        return DinoEmbeddingOutput(
            normalized=params.normalize,
            embedding_dim=int(embedding.shape[0]),
        )
