"""HLS streaming conversion task implementation."""

from pathlib import Path
from typing import Callable, override

from ...common.compute_module import ComputeModule
from ...common.job_storage import JobStorage
from .algo.hls_stream_generator import HLSStreamGenerator, HLSVariant
from .algo.hls_validator import HLSValidator
from .schema import HLSStreamingOutput, HLSStreamingParams


class HLSStreamingTask(ComputeModule[HLSStreamingParams, HLSStreamingOutput]):
    """Compute module for HLS streaming conversion."""

    schema: type[HLSStreamingParams] = HLSStreamingParams

    @property
    @override
    def task_type(self) -> str:
        return "hls_streaming"

    @override
    async def run(
        self,
        job_id: str,
        params: HLSStreamingParams,
        storage: JobStorage,
        progress_callback: Callable[[int], None] | None = None,
    ) -> HLSStreamingOutput:
        input_path = storage.resolve_path(job_id, params.input_path)
        if not input_path.exists():
            raise FileNotFoundError("Input file not found: " + str(input_path))

        # Allocate path for master playlist to ensure output directory exists
        master_playlist_path = storage.allocate_path(
            job_id=job_id,
            relative_path=params.output_path + "/adaptive.m3u8",
        )
        master_playlist = Path(master_playlist_path)
        output_dir = master_playlist.parent

        # Verify directory exists (allocate_path should have created it)
        if not output_dir.exists():
            raise FileNotFoundError(
                f"Output directory was not created by storage: {output_dir}"
            )

        generator = HLSStreamGenerator(
            input_file=str(input_path),
            output_dir=str(output_dir),
        )

        requested_variants = [
            HLSVariant(resolution=v.resolution, bitrate=v.bitrate)
            for v in params.variants
        ]

        _ = generator.addVariants(requested_variants)

        if params.include_original:
            _ = generator.addOriginal()

        validator = HLSValidator(str(master_playlist))
        validation = validator.validate()

        if not validation.is_valid:
            raise RuntimeError("HLS validation failed: " + ", ".join(validation.errors))

        if progress_callback:
            progress_callback(100)

        return HLSStreamingOutput(
            master_playlist=str(master_playlist),
            variants_generated=len(requested_variants),
            total_segments=validation.total_segments,
            include_original=params.include_original,
        )
