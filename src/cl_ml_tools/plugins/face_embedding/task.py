"""Face embedding task implementation."""

import logging
from typing import Callable, TypeAlias, TypedDict, cast, override

import numpy as np
from numpy.typing import NDArray

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from .algo.face_embedder import FaceEmbedder
from .schema import FaceEmbedding, FaceEmbeddingParams

logger = logging.getLogger(__name__)

EmbeddingArray: TypeAlias = NDArray[np.floating]


class FaceEmbeddingFileResult(TypedDict):
    file_path: str
    status: str
    embedding: dict[str, float] | None
    error: str | None


class FaceEmbeddingTask(ComputeModule[FaceEmbeddingParams]):
    """Compute module for generating face embeddings using ONNX model."""

    def __init__(self) -> None:
        super().__init__()
        self._embedder: FaceEmbedder | None = None

    @property
    @override
    def task_type(self) -> str:
        return "face_embedding"

    @override
    def get_schema(self) -> type[BaseJobParams]:
        return FaceEmbeddingParams

    def _get_embedder(self) -> FaceEmbedder:
        if self._embedder is None:
            self._embedder = FaceEmbedder()
            logger.info("Face embedder initialized successfully")
        return self._embedder

    @override
    async def execute(
        self,
        job: Job,
        params: FaceEmbeddingParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        try:
            try:
                embedder = self._get_embedder()
            except Exception as exc:
                logger.error(f"Face embedder initialization failed: {exc}")
                return TaskResult(status = "error", error = (
                        f"Failed to initialize face embedder: {exc}. "
                        "Ensure ONNX Runtime is installed and the model can be downloaded."
                    ))

            file_results: list[FaceEmbeddingFileResult] = []
            total_files: int = len(params.input_paths)

            for index, input_path in enumerate(params.input_paths):
                try:
                    embedding_array, quality_score = cast(
                        tuple[EmbeddingArray, float | None],
                        embedder.embed(
                            image_path=input_path,
                            normalize=params.normalize,
                            compute_quality=True,
                        ),
                    )

                    face_embedding = FaceEmbedding.from_numpy(
                        embedding=embedding_array,
                        quality_score=quality_score,
                    )

                    file_results.append(
                        {
                            "file_path": input_path,
                            "status": "success",
                            "embedding": face_embedding.model_dump(),
                            "error": None,
                        }
                    )

                except FileNotFoundError:
                    logger.error(f"File not found: {input_path}")
                    file_results.append(
                        {
                            "file_path": input_path,
                            "status": "error",
                            "embedding": None,
                            "error": "File not found",
                        }
                    )

                except Exception as exc:
                    logger.error(f"Failed to generate embedding for {input_path}: {exc}")
                    file_results.append(
                        {
                            "file_path": input_path,
                            "status": "error",
                            "embedding": None,
                            "error": str(exc),
                        }
                    )

                if progress_callback:
                    progress = int((index + 1) / total_files * 100)
                    progress_callback(progress)

            all_success: bool = all(r["status"] == "success" for r in file_results)
            any_success: bool = any(r["status"] == "success" for r in file_results)

            if all_success or any_success:
                status = "ok"
                if not all_success:
                    success_count = sum(1 for r in file_results if r["status"] == "success")
                    logger.warning(
                        f"Partial success: {success_count}/{total_files} files processed successfully"
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

        except Exception as exc:
            logger.exception(f"Unexpected error in FaceEmbeddingTask: {exc}")
            return TaskResult(status = "error", error = f"Task failed: {exc}")
