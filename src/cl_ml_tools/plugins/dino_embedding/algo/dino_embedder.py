"""DINOv2 embedding using ONNX model.

Model Source: https://huggingface.co/RoundtTble/dinov2_vits14_onnx
Input: 224x224 RGB images with ImageNet normalization
Output: 384-dimensional CLS token embedding
"""

from pathlib import Path
from typing import Final, cast

import numpy as np
import onnxruntime as ort
from loguru import logger
from numpy.typing import NDArray
from PIL import Image

from ....utils.model_downloader import get_model_downloader

# Model configuration
MODEL_URL: Final[str] = (
    "https://huggingface.co/sefaburak/dinov2-small-onnx/resolve/main/dinov2_vits14.onnx"
)
MODEL_FILENAME: Final[str] = "dinov2_vits14.onnx"
MODEL_SHA256: Final[str | None] = None  # TODO: Add SHA256 hash for verification

# Input configuration
INPUT_SIZE: Final[tuple[int, int]] = (224, 224)  # (height, width)
IMAGENET_MEAN: Final[NDArray[np.float32]] = np.array(
    [0.485, 0.456, 0.406], dtype=np.float32
)
IMAGENET_STD: Final[NDArray[np.float32]] = np.array(
    [0.229, 0.224, 0.225], dtype=np.float32
)


class DinoEmbedder:
    """ONNX-based DINOv2 embedding generator."""

    session: ort.InferenceSession
    input_name: str
    output_name: str

    def __init__(self, model_path: str | Path | None = None) -> None:
        if model_path is None:
            downloader = get_model_downloader()
            logger.info("Downloading DINOv2 model from %s", MODEL_URL)
            model_path = downloader.download(
                url=MODEL_URL,
                filename=MODEL_FILENAME,
                expected_sha256=MODEL_SHA256,
            )
        else:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info("Loading DINOv2 model from %s", model_path)

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
            "DINOv2 model loaded. Input: %s, Output: %s",
            self.input_name,
            self.output_name,
        )

    def preprocess(self, image: Image.Image) -> NDArray[np.float32]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_resized = image.resize(
            (INPUT_SIZE[1], INPUT_SIZE[0]), Image.Resampling.BILINEAR
        )

        img_array: NDArray[np.float32] = (
            np.asarray(image_resized, dtype=np.float32) / 255.0
        )

        img_array = (img_array - IMAGENET_MEAN.reshape(1, 1, 3)) / IMAGENET_STD.reshape(
            1, 1, 3
        )

        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def postprocess(
        self, embedding: NDArray[np.float32], normalize: bool = True
    ) -> NDArray[np.float32]:
        # Remove batch dimension if present
        if embedding.ndim > 1:
            embedding = np.squeeze(embedding)

        # If sequence output, take CLS token
        if embedding.ndim > 1:
            embedding = cast(NDArray[np.float32], embedding[0])

        if normalize:
            norm: float = float(np.linalg.norm(embedding))
            if norm > 0.0:
                embedding = embedding / norm

        return embedding

    def embed(
        self, image_path: str | Path, normalize: bool = True
    ) -> NDArray[np.float32]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(image_path) as image:
            input_array: NDArray[np.float32] = self.preprocess(image)

        outputs = self.session.run([self.output_name], {self.input_name: input_array})
        raw_embedding = cast(NDArray[np.float32], outputs[0])

        embedding: NDArray[np.float32] = self.postprocess(
            raw_embedding, normalize=normalize
        )

        logger.info(
            "Generated DINOv2 embedding for %s: dim=%d",
            image_path,
            int(embedding.shape[0]),
        )

        return embedding
