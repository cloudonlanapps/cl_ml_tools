"""ExifTool wrapper for extracting metadata from media files.

This module provides a wrapper around the ExifTool command-line utility.
ExifTool must be installed separately: https://exiftool.org/
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import TypeAlias, cast

logger = logging.getLogger(__name__)

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]
MetadataDict: TypeAlias = dict[str, JSONValue]


class MetadataExtractor:
    """Wrapper class for extracting metadata from media files using ExifTool."""

    def __init__(self) -> None:
        if not self.is_exiftool_available():
            raise RuntimeError("ExifTool is not installed or not found in PATH.")

    def is_exiftool_available(self) -> bool:
        try:
            result = subprocess.run(
                ["exiftool", "-ver"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            logger.debug(f"ExifTool version {result.stdout.strip()} found")
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def extract_metadata(self, filepath: str | Path, tags: list[str]) -> MetadataDict:
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not tags:
            return {}

        tag_args = [f"-{tag}" for tag in tags]

        try:
            result = subprocess.run(
                ["exiftool", "-n", "-j", *tag_args, str(path)],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            parsed = cast(list[MetadataDict], json.loads(result.stdout))
            return parsed[0] if parsed else {}

        except subprocess.CalledProcessError as exc:
            stderr = cast(str, exc.stderr) if exc.stderr is not None else ""  # pyright: ignore[reportAny]
            logger.error(f"ExifTool failed for {path}: {stderr}")
            return {}

        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            return {}

    def extract_metadata_all(self, filepath: str | Path) -> MetadataDict:
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            result = subprocess.run(
                ["exiftool", "-G", "-n", "-j", str(path)],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            parsed = cast(list[MetadataDict], json.loads(result.stdout))
            return parsed[0] if parsed else {}

        except subprocess.CalledProcessError as exc:
            stderr = cast(str, exc.stderr) if exc.stderr is not None else ""  # pyright: ignore[reportAny]
            logger.error(f"ExifTool failed for {path}: {stderr}")
            return {}

        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            return {}
