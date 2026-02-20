"""Hash computation task implementation."""

import json
import time
from io import BytesIO
from typing import Callable, override
import magic

from ...common.compute_module import ComputeModule
from ...common.job_storage import JobStorage
from ...utils.media_types import MediaType, determine_media_type
from .algo.generic import sha512hash_generic
from .algo.image import sha512hash_image
from .algo.md5 import get_md5_hexdigest
from .algo.video import sha512hash_video2
from .schema import HashOutput, HashParams


class HashTask(ComputeModule[HashParams, HashOutput]):
    """Compute module for file hashing with media-type detection."""

    schema: type[HashParams] = HashParams

    @property
    @override
    def task_type(self) -> str:
        return "hash"

    @override
    async def run(
        self,
        job_id: str,
        params: HashParams,
        storage: JobStorage,
        progress_callback: Callable[[int], None] | None = None,
    ) -> HashOutput:
        input_path = storage.resolve_path(job_id, params.input_path)

        if not input_path.exists():
            raise FileNotFoundError("Input file not found: " + str(input_path))

        file_bytes = input_path.read_bytes()
        bytes_io = BytesIO(file_bytes)

        # Get MIME type robustly
        try:
            mime_type_str, media_type = determine_mime(bytes_io)
        except Exception as e:
            logger.warning(f"MIME detection failed in HashTask: {e}")
            media_type = MediaType.FILE

        _ = bytes_io.seek(0)
        start = time.time()

        if params.algorithm == "md5":
            hash_value = get_md5_hexdigest(bytes_io)
            algorithm_used = "md5"

        elif media_type == MediaType.IMAGE:
            hash_value, _ = sha512hash_image(bytes_io)
            algorithm_used = "sha512_image"

        elif media_type == MediaType.VIDEO:
            hash_bytes = sha512hash_video2(bytes_io)
            hash_value = hash_bytes.hex()
            algorithm_used = "sha512_video"

        else:
            hash_value, _ = sha512hash_generic(bytes_io)
            algorithm_used = "sha512_generic"

        process_time = time.time() - start

        payload = {
            "hash_value": hash_value,
            "algorithm": algorithm_used,
            "media_type": media_type.value,
            "process_time": process_time,
        }

        path = storage.allocate_path(
            job_id=job_id,
            relative_path=params.output_path,
        )

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        if progress_callback:
            progress_callback(100)

        return HashOutput(
            media_type=media_type.value,
        )
