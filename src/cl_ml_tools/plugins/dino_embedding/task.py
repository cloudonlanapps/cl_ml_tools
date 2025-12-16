"""DINOv2 embedding task implementation."""

import logging
from typing import Callable, override

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from .algo.dino_embedder import DinoEmbedder
from .schema import DinoEmbedding, DinoEmbeddingParams, DinoEmbeddingResult

logger = logging.getLogger(__name__)


class DinoEmbeddingTask(ComputeModule[DinoEmbeddingParams]):
    """Compute module for generating DINOv2 embeddings using ONNX model."""

    def __init__(self) -> None:
        super().__init__()
        self._embedder: DinoEmbedder | None = None

    @property
    @override
    def task_type(self) -> str:
        return "dino_embedding"

    @override
    def get_schema(self) -> type[BaseJobParams]:
        return DinoEmbeddingParams

    def _get_embedder(self) -> DinoEmbedder:
        if self._embedder is None:
            try:
                self._embedder = DinoEmbedder()
                logger.info("DINOv2 embedder initialized successfully")
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to initialize DINOv2 embedder: %s", exc)
                raise
        return self._embedder

    @override
    async def execute(
        self,
        job: Job,
        params: DinoEmbeddingParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        try:
            try:
                embedder = self._get_embedder()
            except Exception as exc:  # noqa: BLE001
                return TaskResult(status = "error", error = (
                        f"Failed to initialize DINOv2 embedder: {exc}. "
                        "Ensure ONNX Runtime is installed and the model can be downloaded."
                    ))

            file_results: list[dict[str, object]] = []
            total_files: int = len(params.input_paths)

            for index, input_path in enumerate(params.input_paths):
                try:
                    embedding_array = embedder.embed(
                        image_path=input_path,
                        normalize=params.normalize,
                    )

                    dino_embedding = DinoEmbedding.from_numpy(embedding_array)

                    result = DinoEmbeddingResult(
                        file_path=input_path,
                        embedding=dino_embedding,
                        status="success",
                    )
                    file_results.append(result.model_dump())

                except FileNotFoundError:
                    result = DinoEmbeddingResult(
                        file_path=input_path,
                        embedding=None,
                        status="error",
                        error="File not found",
                    )
                    file_results.append(result.model_dump())

                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to generate DINOv2 embedding for %s: %s",
                        input_path,
                        exc,
                    )
                    result = DinoEmbeddingResult(
                        file_path=input_path,
                        embedding=None,
                        status="error",
                        error=str(exc),
                    )
                    file_results.append(result.model_dump())

                if progress_callback:
                    progress: int = int((index + 1) / total_files * 100)
                    progress_callback(progress)

            all_success: bool = all(r.get("status") == "success" for r in file_results)
            any_success: bool = any(r.get("status") == "success" for r in file_results)

            if all_success:
                status: str = "ok"
            elif any_success:
                status = "ok"
                success_count: int = sum(1 for r in file_results if r.get("status") == "success")
                logger.warning(
                    "Partial success: %d/%d files processed successfully",
                    success_count,
                    total_files,
                )
            else:
                return TaskResult(status = "error", task_output = {
                        "files": file_results,
                        "total_files": total_files,
                    }, error = "Failed to generate embeddings for all files")

            return TaskResult(status = status, task_output = {
                    "files": file_results,
                    "total_files": total_files,
                    "normalize": params.normalize,
                })

        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in DinoEmbeddingTask: %s", exc)
            return TaskResult(status = "error", error = f"Task failed: {exc}")
