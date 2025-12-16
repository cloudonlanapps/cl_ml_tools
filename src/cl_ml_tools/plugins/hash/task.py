"""Hash computation task implementation."""

import time
from io import BytesIO
from pathlib import Path
from typing import Callable, override

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from ...utils.media_types import MediaType, determine_media_type
from .algo.generic import sha512hash_generic
from .algo.image import sha512hash_image
from .algo.md5 import get_md5_hexdigest
from .algo.video import sha512hash_video2
from .schema import HashParams


class HashTask(ComputeModule[HashParams]):
    """Compute module for file hashing with media-type detection."""

    @property
    @override
    def task_type(self) -> str:
        return "hash"

    @override
    def get_schema(self) -> type[BaseJobParams]:
        return HashParams

    @override
    async def execute(
        self,
        job: Job,
        params: HashParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        try:
            file_results: list[dict[str, object]] = []
            total_files = len(params.input_paths)

            for index, input_path in enumerate(params.input_paths):
                # Read file into BytesIO
                file_path = Path(input_path)
                file_bytes = file_path.read_bytes()
                bytes_io = BytesIO(file_bytes)

                # Determine media type using python-magic
                import magic

                mime = magic.Magic(mime=True)
                file_type = mime.from_buffer(file_bytes)
                media_type = determine_media_type(bytes_io, file_type)

                # Route to appropriate hash function
                hash_value: str
                process_time: float

                if params.algorithm == "md5":
                    # MD5 algorithm (any file type)
                    _ = bytes_io.seek(0)
                    hash_value = get_md5_hexdigest(bytes_io)
                    process_time = 0.0  # MD5 algo doesn't track time
                    algorithm_used = "md5"

                elif media_type == MediaType.IMAGE:
                    # SHA-512 for images (uses PIL)
                    _ = bytes_io.seek(0)
                    hash_value, process_time = sha512hash_image(bytes_io)
                    algorithm_used = "sha512_image"

                elif media_type == MediaType.VIDEO:
                    # SHA-512 for videos (I-frames only)
                    _ = bytes_io.seek(0)
                    start_time = time.time()
                    hash_bytes = sha512hash_video2(bytes_io)
                    hash_value = hash_bytes.hex()
                    end_time = time.time()
                    process_time = end_time - start_time
                    algorithm_used = "sha512_video"

                else:
                    # Generic SHA-512 for TEXT, AUDIO, FILE, URL
                    _ = bytes_io.seek(0)
                    hash_value, process_time = sha512hash_generic(bytes_io)
                    algorithm_used = "sha512_generic"

                # Collect result for this file
                file_results.append(
                    {
                        "file_path": input_path,
                        "media_type": media_type.value,
                        "hash_value": hash_value,
                        "algorithm_used": algorithm_used,
                        "process_time": process_time,
                    }
                )

                # Report progress
                if progress_callback:
                    progress = int((index + 1) / total_files * 100)
                    progress_callback(progress)

            return TaskResult(status = "ok", task_output = {
                    "files": file_results,
                    "total_files": total_files,
                })

        except ImportError as e:
            return TaskResult(status = "error", error = f"Missing dependency: {e}. Install with: pip install cl_ml_tools[compute]")
        except FileNotFoundError as e:
            return TaskResult(status = "error", error = f"Input file not found: {e}")
        except Exception as e:
            return TaskResult(status = "error", error = f"Hash computation failed: {e}")
