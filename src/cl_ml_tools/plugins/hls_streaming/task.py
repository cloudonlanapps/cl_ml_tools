"""HLS streaming conversion task implementation."""

from pathlib import Path
from typing import Callable, override

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from .algo.hls_stream_generator import HLSStreamGenerator, HLSVariant
from .algo.hls_validator import HLSValidator, ValidationResult
from .schema import HLSConversionResult, HLSStreamingParams, HLSStreamingTaskOutput


class HLSStreamingTask(ComputeModule[HLSStreamingParams]):
    """Compute module for HLS streaming conversion."""

    @property
    @override
    def task_type(self) -> str:
        return "hls_streaming"

    @override
    def get_schema(self) -> type[BaseJobParams]:
        return HLSStreamingParams

    @override
    async def execute(
        self,
        job: Job,
        params: HLSStreamingParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        try:
            results: list[HLSConversionResult] = []
            total_files = len(params.input_paths)

            for index, input_path in enumerate(params.input_paths):
                output_dir = params.output_paths[index]

                # Create output directory
                Path(output_dir).mkdir(parents=True, exist_ok=True)

                # Initialize HLS generator
                generator = HLSStreamGenerator(input_file=input_path, output_dir=output_dir)

                # Convert variant configs to HLSVariant objects
                requested_variants = [
                    HLSVariant(resolution=v.resolution, bitrate=v.bitrate) for v in params.variants
                ]

                # Generate HLS streams
                _ = generator.addVariants(requested_variants)

                # Add original if requested
                if params.include_original:
                    _ = generator.addOriginal()

                # Validate output
                master_playlist = Path(output_dir) / "adaptive.m3u8"
                validator = HLSValidator(str(master_playlist))
                validation: ValidationResult = validator.validate()

                if not validation.is_valid:
                    return TaskResult(status = "error", error = f"HLS validation failed: {', '.join(validation.errors)}")

                # Create Pydantic result object (not dict!)
                result = HLSConversionResult(
                    input_file=input_path,
                    output_dir=output_dir,
                    master_playlist=str(master_playlist),
                    variants_generated=len(requested_variants),
                    total_segments=validation.total_segments,
                    include_original=params.include_original,
                )
                results.append(result)

                # Report progress
                if progress_callback:
                    progress = int((index + 1) / total_files * 100)
                    progress_callback(progress)

            # Create Pydantic task output (not dict!)
            task_output = HLSStreamingTaskOutput(
                files=results,
                total_files=total_files,
            )

            return TaskResult(status = "ok", task_output = task_output.model_dump())

        except FileNotFoundError as e:
            return TaskResult(status = "error", error = f"Input file not found: {e}")
        except Exception as e:
            return TaskResult(status = "error", error = f"HLS conversion failed: {e}")
