"""HLS streaming conversion task implementation."""

from pathlib import Path
from typing import Callable, override
from loguru import logger

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
        # Resolve Input Path
        if params.input_absolute_path:
            input_path = Path(params.input_absolute_path)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
        else:
            input_path = storage.resolve_path(job_id, params.input_path)
            if not input_path.exists():
                raise FileNotFoundError("Input file not found: " + str(input_path))

        # Resolve Output Directory
        if params.output_absolute_path:
            output_dir = Path(params.output_absolute_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            master_playlist_path = output_dir / "adaptive.m3u8"
        else:
            # Allocate path via storage (creates directories)
            master_playlist_path_str = storage.allocate_path(
                job_id=job_id,
                relative_path=params.output_path + "/adaptive.m3u8",
            )
            master_playlist_path = Path(master_playlist_path_str)
            output_dir = master_playlist_path.parent

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

        def manifest_ready_callback():
            if progress_callback:
                logger.info(f"HLS Manifest ready for job {job_id}, signaling progress 1%")
                progress_callback(1)

        _ = generator.addVariants(
            requested_variants, on_ready=manifest_ready_callback
        )

        if params.include_original:
            _ = generator.addOriginal(on_ready=manifest_ready_callback)

        validator = HLSValidator(str(master_playlist_path))
        validation = validator.validate()

        if not validation.is_valid:
            raise RuntimeError("HLS validation failed: " + ", ".join(validation.errors))

        if progress_callback:
            progress_callback(100)

        return HLSStreamingOutput(
            master_playlist=str(master_playlist_path),
            variants_generated=len(requested_variants),
            total_segments=validation.total_segments,
            include_original=params.include_original,
        )
