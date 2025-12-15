"""Schema definitions for HLS streaming conversion plugin."""

from pydantic import BaseModel, Field, field_validator

from ...common.schemas import BaseJobParams


class VariantConfig(BaseModel):
    """Configuration for a single HLS variant.

    Users can fully customize resolution and bitrate for each variant.
    """

    resolution: int | None = Field(
        default=None,
        description="Target height in pixels (e.g., 720, 480, 240). None for original.",
    )
    bitrate: int | None = Field(
        default=None,
        ge=100,
        description="Target bitrate in kbps (e.g., 3500, 1500, 800)",
    )


class HLSConversionResult(BaseModel):
    """Result for a single HLS conversion (Pydantic model, not dict)."""

    input_file: str = Field(description="Path to input video file")
    output_dir: str = Field(description="Directory containing HLS output")
    master_playlist: str = Field(description="Path to master M3U8 playlist")
    variants_generated: int = Field(description="Number of variants created")
    total_segments: int = Field(description="Total TS segments across all variants")
    include_original: bool = Field(
        description="Whether original quality was included"
    )


class HLSStreamingTaskOutput(BaseModel):
    """Task output structure (Pydantic model, not dict)."""

    files: list[HLSConversionResult] = Field(
        description="Conversion results for each input file"
    )
    total_files: int = Field(description="Total number of files processed")


class HLSStreamingParams(BaseJobParams):
    """Parameters for HLS streaming conversion task."""

    variants: list[VariantConfig] = Field(
        default_factory=lambda: [
            VariantConfig(resolution=720, bitrate=3500),
            VariantConfig(resolution=480, bitrate=1500),
            VariantConfig(resolution=240, bitrate=800),
        ],
        description="List of quality variants to generate (fully customizable)",
    )

    include_original: bool = Field(
        default=False, description="Include original quality without re-encoding"
    )

    @field_validator("variants")
    @classmethod
    def validate_variants_not_empty(
        cls, v: list[VariantConfig]
    ) -> list[VariantConfig]:
        """Ensure at least one variant is specified."""
        if len(v) == 0:
            raise ValueError("At least one variant must be specified")
        return v
