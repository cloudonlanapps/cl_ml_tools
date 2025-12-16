"""EXIF metadata extraction parameters and output schemas."""

from typing import Any

from pydantic import BaseModel, Field

from ...common.schemas import BaseJobParams


class ExifParams(BaseJobParams):
    """Parameters for EXIF metadata extraction task.

    Attributes:
        input_paths: List of absolute paths to input media files
        output_paths: Not used for EXIF extraction (metadata returned in task_output)
        tags: List of EXIF tags to extract (e.g., ["Make", "Model", "DateTimeOriginal"])
              If empty, all available EXIF tags will be extracted
    """

    tags: list[str] = Field(
        default_factory=list,
        description="List of EXIF tags to extract. Empty list extracts all tags.",
    )


class ExifMetadata(BaseModel):
    """Typed EXIF metadata output model.

    This model contains commonly used EXIF fields with proper typing.
    The raw_metadata field contains the complete metadata dictionary.
    """

    # Core identification fields
    make: str | None = Field(default=None, description="Camera manufacturer (e.g., Canon, Nikon)")
    model: str | None = Field(default=None, description="Camera model (e.g., Canon EOS R5)")

    # DateTime fields
    date_time_original: str | None = Field(
        default=None, description="Date/time when photo was taken (EXIF:DateTimeOriginal)"
    )
    create_date: str | None = Field(
        default=None, description="File creation date (EXIF:CreateDate)"
    )

    # Image properties
    image_width: int | None = Field(default=None, description="Image width in pixels")
    image_height: int | None = Field(default=None, description="Image height in pixels")
    orientation: int | None = Field(
        default=None, description="Image orientation (1=normal, 3=180°, 6=90°CW, 8=270°CW)"
    )

    # Camera settings
    iso: int | None = Field(default=None, description="ISO speed (e.g., 100, 400, 1600)")
    f_number: float | None = Field(default=None, description="F-stop/aperture (e.g., 2.8, 5.6)")
    exposure_time: str | None = Field(
        default=None, description="Shutter speed (e.g., 1/125, 1/1000)"
    )
    focal_length: float | None = Field(default=None, description="Focal length in mm")

    # GPS coordinates
    gps_latitude: float | None = Field(default=None, description="GPS latitude in decimal degrees")
    gps_longitude: float | None = Field(
        default=None, description="GPS longitude in decimal degrees"
    )
    gps_altitude: float | None = Field(default=None, description="GPS altitude in meters")

    # Software/processing
    software: str | None = Field(
        default=None, description="Software used to create/modify the image"
    )

    # Raw metadata dictionary (complete EXIF data)
    raw_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Complete EXIF metadata dictionary from ExifTool"
    )

    @classmethod
    def from_raw_metadata(cls, raw_meta: dict[str, Any]) -> "ExifMetadata":
        """Create ExifMetadata from raw ExifTool output.

        Args:
            raw_meta: Raw metadata dictionary from ExifTool

        Returns:
            ExifMetadata instance with typed fields populated
        """
        return cls(
            make=raw_meta.get("Make"),
            model=raw_meta.get("Model"),
            date_time_original=raw_meta.get("DateTimeOriginal"),
            create_date=raw_meta.get("CreateDate"),
            image_width=raw_meta.get("ImageWidth"),
            image_height=raw_meta.get("ImageHeight"),
            orientation=raw_meta.get("Orientation"),
            iso=raw_meta.get("ISO"),
            f_number=raw_meta.get("FNumber"),
            exposure_time=raw_meta.get("ExposureTime"),
            focal_length=raw_meta.get("FocalLength"),
            gps_latitude=raw_meta.get("GPSLatitude"),
            gps_longitude=raw_meta.get("GPSLongitude"),
            gps_altitude=raw_meta.get("GPSAltitude"),
            software=raw_meta.get("Software"),
            raw_metadata=raw_meta,
        )
