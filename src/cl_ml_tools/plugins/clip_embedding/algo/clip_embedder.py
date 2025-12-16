"""MobileCLIP embedding using ONNX model."""

import logging
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from PIL import Image

from ....utils.model_downloader import get_model_downloader

logger = logging.getLogger(__name__)

MODEL_URL = "https://huggingface.co/apple/MobileCLIP-S2-onnx/resolve/main/image_encoder.onnx"
MODEL_FILENAME = "mobileclip_s2_image_encoder.onnx"
MODEL_SHA256: str | None = None

INPUT_SIZE = (256, 256)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class ClipEmbedder:
    """ONNX-based MobileCLIP image embedding generator."""

    session: ort.InferenceSession
    input_name: str
    output_name: str

    def __init__(self, model_path: str | Path | None = None) -> None:
        if model_path is None:
            downloader = get_model_downloader()
            logger.info("Downloading MobileCLIP model")
            model_path = downloader.download(
                url=MODEL_URL,
                filename=MODEL_FILENAME,
                expected_sha256=MODEL_SHA256,
            )
        else:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(model_path)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image: Image.Image) -> NDArray[np.float32]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize(
            (INPUT_SIZE[1], INPUT_SIZE[0]),
            Image.Resampling.BICUBIC,
        )

        img = np.asarray(image, dtype=np.float32) / 255.0
        img = (img - CLIP_MEAN) / CLIP_STD
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img

    def postprocess(
        self,
        embedding: NDArray[np.float32],
        normalize: bool = True,
    ) -> NDArray[np.float32]:
        embedding = np.squeeze(embedding)

        if embedding.ndim != 1:
            embedding = embedding.flatten()

        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = (embedding / norm).astype(np.float32)

        return embedding

    def embed(
        self,
        image_path: str | Path,
        normalize: bool = True,
    ) -> NDArray[np.float32]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        image = Image.open(image_path)
        input_array = self.preprocess(image)

        output = cast(
            NDArray[np.float32],
            self.session.run(
                [self.output_name],
                {self.input_name: input_array},
            )[0],
        )

        embedding = self.postprocess(output, normalize=normalize)

        logger.info("Generated embedding: dim=%d", embedding.shape[0])
        return embedding
