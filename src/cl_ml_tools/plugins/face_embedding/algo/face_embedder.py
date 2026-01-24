"""Face embedding using ONNX model (ArcFace).

Model Source: https://huggingface.co/garavv/arcface-onnx
Input: 112x112 RGB face images
Output: 512-dimensional embedding
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort
from loguru import logger
from numpy.typing import NDArray
from PIL import Image
from scipy import signal

from ....utils.model_downloader import get_model_downloader
from ....utils.profiling import timed

# Model configuration
MODEL_URL = (
    "https://huggingface.co/onnx-community/arcface-onnx/resolve/main/arcface.onnx"
)
MODEL_FILENAME = "arcface_face_embedding.onnx"
MODEL_SHA256 = None  # TODO: Add SHA256 hash for verification

# Expected input shape for ArcFace
INPUT_SIZE: tuple[int, int] = (112, 112)  # (height, width)


class FaceEmbedder:
    """ONNX-based face embedding generator using ArcFace model."""

    session: ort.InferenceSession
    input_name: str
    output_name: str

    def __init__(self, model_path: str | Path | None = None) -> None:
        if model_path is None:
            downloader = get_model_downloader()
            logger.info(f"Downloading face embedding model from {MODEL_URL}")
            model_path = downloader.download(
                url=MODEL_URL,
                filename=MODEL_FILENAME,
                expected_sha256=MODEL_SHA256,
            )
        else:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading face embedding model from {model_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(
            f"Model loaded. Input: {self.input_name}, Output: {self.output_name}"
        )

    @timed
    def preprocess(self, image: Image.Image) -> NDArray[np.float32]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_resized = image.resize(
            (INPUT_SIZE[1], INPUT_SIZE[0]),
            Image.Resampling.BILINEAR,
        )

        img_array: NDArray[np.float32] = (
            np.asarray(image_resized, dtype=np.float32) / 255.0
        )

        # PIL already gives us (H, W, C), model expects (batch, H, W, C)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    @timed
    def postprocess(
        self, embedding: NDArray[np.float32], normalize: bool = True
    ) -> NDArray[np.float32]:
        if embedding.ndim > 1:
            embedding = embedding.squeeze(axis=0)

        if normalize:
            norm = float(np.linalg.norm(embedding))
            if norm > 0.0:
                embedding = embedding / norm

        return embedding

    @timed
    def compute_quality_score(self, image: Image.Image) -> float:
        img_gray = image.convert("L")
        img_array: NDArray[np.float32] = np.asarray(img_gray, dtype=np.float32)

        kernel: NDArray[np.float32] = np.array(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )

        # Use vectorized convolution instead of nested loops
        lap_img = signal.convolve2d(img_array, kernel, mode="same", boundary="symm")

        variance = float(np.var(lap_img))
        quality = min(variance / 100.0, 1.0)

        return quality

    @timed
    def embed(
        self,
        image_path: str | Path,
        normalize: bool = True,
        compute_quality: bool = True,
    ) -> tuple[NDArray[np.float32], float | None]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(image_path) as image:
            input_array = self.preprocess(image)

            @timed
            def _inference_run():
                return cast(
                    NDArray[np.float32],
                    self.session.run(
                        [self.output_name],
                        {self.input_name: input_array},
                    )[0],
                )

            raw_output = _inference_run()

            embedding = self.postprocess(raw_output, normalize=normalize)

            quality_score: float | None = None
            if compute_quality:
                try:
                    quality_score = self.compute_quality_score(image)
                except (ValueError, RuntimeError, OSError) as exc:
                    logger.warning(f"Failed to compute quality score: {exc}")

        logger.info(
            f"Generated embedding for {image_path}: dim={embedding.shape[0]}, quality={quality_score}"
        )

        return embedding, quality_score
