"""EXIF metadata extraction parameters and output schemas."""

from typing import TypeAlias

from pydantic import BaseModel, Field

from ...common.schemas import BaseJobParams

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]


def _as_str(value: JSONValue) -> str | None:
    return value if isinstance(value, str) else None


def _as_int(value: JSONValue) -> int | None:
    return value if isinstance(value, int) else None


def _as_float(value: JSONValue) -> float | None:
    return value if isinstance(value, (int, float)) else None


class ExifParams(BaseJobParams):
    """Parameters for EXIF metadata extraction task."""

    tags: list[str] = Field(
        default_factory=list,
        description="List of EXIF tags to extract. Empty list extracts all tags.",
    )


class ExifMetadata(BaseModel):
    """Typed EXIF metadata output model."""

    make: str | None = None
    model: str | None = None

    date_time_original: str | None = None
    create_date: str | None = None

    image_width: int | None = None
    image_height: int | None = None
    orientation: int | None = None

    iso: int | None = None
    f_number: float | None = None
    exposure_time: str | None = None
    focal_length: float | None = None

    gps_latitude: float | None = None
    gps_longitude: float | None = None
    gps_altitude: float | None = None

    software: str | None = None

    raw_metadata: dict[str, JSONValue] = Field(default_factory=dict)

    @classmethod
    def from_raw_metadata(cls, raw_meta: dict[str, JSONValue]) -> "ExifMetadata":
        return cls(
            make=_as_str(raw_meta.get("Make")),
            model=_as_str(raw_meta.get("Model")),
            date_time_original=_as_str(raw_meta.get("DateTimeOriginal")),
            create_date=_as_str(raw_meta.get("CreateDate")),
            image_width=_as_int(raw_meta.get("ImageWidth")),
            image_height=_as_int(raw_meta.get("ImageHeight")),
            orientation=_as_int(raw_meta.get("Orientation")),
            iso=_as_int(raw_meta.get("ISO")),
            f_number=_as_float(raw_meta.get("FNumber")),
            exposure_time=_as_str(raw_meta.get("ExposureTime")),
            focal_length=_as_float(raw_meta.get("FocalLength")),
            gps_latitude=_as_float(raw_meta.get("GPSLatitude")),
            gps_longitude=_as_float(raw_meta.get("GPSLongitude")),
            gps_altitude=_as_float(raw_meta.get("GPSAltitude")),
            software=_as_str(raw_meta.get("Software")),
            raw_metadata=raw_meta,
        )
