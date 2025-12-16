"""ExifTool wrapper for extracting metadata from media files.

This module provides a wrapper around the ExifTool command-line utility.
ExifTool must be installed separately: https://exiftool.org/
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Wrapper class for extracting metadata from media files using ExifTool.

    Requires ExifTool to be installed and available in PATH.
    Install from: https://exiftool.org/
    """

    def __init__(self):
        """Initialize the MetadataExtractor and check for ExifTool availability.

        Raises:
            RuntimeError: If ExifTool is not installed or not found in PATH
        """
        if not self.is_exiftool_available():
            raise RuntimeError("ExifTool is not installed or not found in PATH.")

    def is_exiftool_available(self) -> bool:
        """Check if ExifTool is available in the system.

        Returns:
            True if ExifTool is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["exiftool", "-ver"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            version = result.stdout.strip()
            logger.debug(f"ExifTool version {version} found")
            return True
        except FileNotFoundError:
            logger.warning("ExifTool not found in PATH")
            return False
        except subprocess.CalledProcessError as e:
            logger.warning(f"ExifTool check failed: {e}")
            return False
        except subprocess.TimeoutExpired:
            logger.warning("ExifTool check timed out")
            return False

    def extract_metadata(
        self, filepath: str | Path, tags: list[str]
    ) -> dict[str, Any]:
        """Extract specific metadata tags from a media file using ExifTool.

        Args:
            filepath: Path to the media file
            tags: List of metadata tags to extract (e.g., ["Make", "Model", "DateTimeOriginal"])

        Returns:
            Extracted metadata as a dictionary. Returns empty dict on error.

        Raises:
            FileNotFoundError: If the file does not exist
        """
        filepath = Path(filepath)

        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")

        if not tags:
            logger.error("No tags provided for metadata extraction")
            return {}

        # Format tags for ExifTool
        tag_args = [f"-{tag}" for tag in tags]

        try:
            result = subprocess.run(
                ["exiftool", "-n", "-j"] + tag_args + [str(filepath)],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            metadata = json.loads(result.stdout)

            if not metadata:
                logger.warning(f"No metadata found in {filepath}")
                return {}

            # ExifTool returns a list; we take the first result
            return metadata[0]

        except subprocess.CalledProcessError as e:
            logger.error(f"ExifTool failed for {filepath}: {e.stderr}")
            return {}

        except subprocess.TimeoutExpired:
            logger.error(f"ExifTool timed out for {filepath}")
            return {}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse ExifTool JSON output for {filepath}: {e}")
            return {}

    def extract_metadata_all(self, filepath: str | Path) -> dict[str, Any]:
        """Extract all available metadata from a media file using ExifTool.

        Args:
            filepath: Path to the media file

        Returns:
            Complete metadata dictionary. Returns empty dict on error.

        Raises:
            FileNotFoundError: If the file does not exist
        """
        filepath = Path(filepath)

        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            # -G: Include group names, -n: Numeric output, -j: JSON format
            result = subprocess.run(
                ["exiftool", "-G", "-n", "-j", str(filepath)],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            metadata = json.loads(result.stdout)

            if not metadata:
                logger.warning(f"No metadata found in {filepath}")
                return {}

            return metadata[0]

        except subprocess.CalledProcessError as e:
            logger.error(f"ExifTool failed for {filepath}: {e.stderr}")
            return {}

        except subprocess.TimeoutExpired:
            logger.error(f"ExifTool timed out for {filepath}")
            return {}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse ExifTool JSON output for {filepath}: {e}")
            return {}
