"""MobileCLIP embedding task implementation."""

import logging
from typing import Callable, override

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from .algo.clip_embedder import ClipEmbedder
from .schema import ClipEmbedding, ClipEmbeddingParams, ClipEmbeddingResult

logger = logging.getLogger(__name__)


class ClipEmbeddingTask(ComputeModule[ClipEmbeddingParams]):
    """Compute module for generating MobileCLIP embeddings using ONNX model."""

    def __init__(self) -> None:
        """Initialize MobileCLIP embedding task."""
        super().__init__()
        self._embedder: ClipEmbedder | None = None

    @property
    @override
    def task_type(self) -> str:
        return "clip_embedding"

    @override
    def get_schema(self) -> type[BaseJobParams]:
        return ClipEmbeddingParams

    def _get_embedder(self) -> ClipEmbedder:
        """Get or create MobileCLIP embedder instance (lazy loading)."""
        if self._embedder is None:
            try:
                self._embedder = ClipEmbedder()
                logger.info("MobileCLIP embedder initialized successfully")
            except Exception as exc:
                logger.error("Failed to initialize MobileCLIP embedder", exc_info=exc)
                raise

        return self._embedder

    @override
    async def execute(
        self,
        job: Job,
        params: ClipEmbeddingParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        """Generate MobileCLIP embeddings for input images."""
        try:
            try:
                embedder = self._get_embedder()
            except Exception as exc:
                return TaskResult(status = "error", error = (
                        f"Failed to initialize MobileCLIP embedder: {exc}. "
                        "Ensure ONNX Runtime is installed and the model is available."
                    ))

            file_results: list[ClipEmbeddingResult] = []
            total_files = len(params.input_paths)

            for index, input_path in enumerate(params.input_paths):
                try:
                    embedding_array = embedder.embed(
                        image_path=input_path,
                        normalize=params.normalize,
                    )

                    clip_embedding = ClipEmbedding.from_numpy(
                        embedding=embedding_array,
                        normalized=params.normalize,
                    )

                    result = ClipEmbeddingResult(
                        file_path=input_path,
                        embedding=clip_embedding,
                        status="success",
                        error=None,
                    )

                except FileNotFoundError:
                    logger.error("File not found: %s", input_path)
                    result = ClipEmbeddingResult(
                        file_path=input_path,
                        embedding=None,
                        status="error",
                        error="File not found",
                    )

                except Exception as exc:
                    logger.error(
                        "Failed to generate MobileCLIP embedding for %s",
                        input_path,
                        exc_info=exc,
                    )
                    result = ClipEmbeddingResult(
                        file_path=input_path,
                        embedding=None,
                        status="error",
                        error=str(exc),
                    )

                file_results.append(result)

                if progress_callback:
                    progress = int((index + 1) / total_files * 100)
                    progress_callback(progress)

            all_success = all(r.status == "success" for r in file_results)
            any_success = any(r.status == "success" for r in file_results)

            if not any_success:
                return TaskResult(status = "error", task_output = {
                        "files": [r.model_dump() for r in file_results],
                        "total_files": total_files,
                    }, error = "Failed to generate embeddings for all files")

            if not all_success:
                logger.warning(
                    "Partial success: %d/%d files processed successfully",
                    sum(1 for r in file_results if r.status == "success"),
                    total_files,
                )

            return TaskResult(status = "ok", task_output = {
                    "files": [r.model_dump() for r in file_results],
                    "total_files": total_files,
                    "normalize": params.normalize,
                })

        except Exception as exc:
            logger.exception("Unexpected error in ClipEmbeddingTask")
            return TaskResult(status = "error", error = f"Task failed: {exc}")
