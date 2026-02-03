#!/usr/bin/env python3
"""Generate test images with known EXIF metadata using exiftool.

This script creates test images with specific EXIF metadata for testing
the EXIF extraction plugin. Requires exiftool to be installed.
"""

import hashlib
import subprocess
from pathlib import Path

from PIL import Image

import os
# Configuration
TESTS_DIR = Path(__file__).parent
TEST_MEDIA_DIR = Path(
    os.getenv("TEST_VECTORS_DIR", str(Path.home() / "cl_server_test_media"))
)
TARGET_DIR = TEST_MEDIA_DIR / "exif_generated"
MANIFEST_FILE = TESTS_DIR / "MANIFEST.md5"


def check_exiftool_installed() -> bool:
    """Check if exiftool is installed and available."""
    try:
        result = subprocess.run(
            ["exiftool", "-ver"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def calculate_md5(file_path: Path) -> str:
    """Calculate MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def create_blank_image(output_path: Path, size: tuple[int, int] = (800, 600)) -> None:
    """Create a blank image for EXIF injection."""
    img = Image.new("RGB", size, color=(100, 150, 200))
    img.save(output_path, "JPEG", quality=95)


def generate_gps_image() -> Path:
    """
    Generate image with GPS coordinates.

    GPS: 37.7749° N, 122.4194° W (San Francisco)
    """
    output_path = TARGET_DIR / "with_gps.jpg"
    create_blank_image(output_path)

    _ = subprocess.run(
        [
            "exiftool",
            "-overwrite_original",
            "-GPSLatitude=37.7749",
            "-GPSLatitudeRef=N",
            "-GPSLongitude=122.4194",
            "-GPSLongitudeRef=W",
            "-GPSAltitude=10",
            "-GPSAltitudeRef=0",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )

    return output_path


def generate_camera_settings_image() -> Path:
    """
    Generate image with camera settings.

    Settings: ISO 400, Aperture f/2.8, Shutter 1/100
    """
    output_path = TARGET_DIR / "with_camera.jpg"
    create_blank_image(output_path)

    _ = subprocess.run(
        [
            "exiftool",
            "-overwrite_original",
            "-Make=Canon",
            "-Model=Canon EOS R5",
            "-ISO=400",
            "-FNumber=2.8",
            "-ExposureTime=1/100",
            "-FocalLength=50",
            "-LensModel=RF 50mm F1.8 STM",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )

    return output_path


def generate_datetime_image() -> Path:
    """
    Generate image with specific datetime.

    DateTime: 2024-01-15 14:30:00
    """
    output_path = TARGET_DIR / "with_datetime.jpg"
    create_blank_image(output_path)

    _ = subprocess.run(
        [
            "exiftool",
            "-overwrite_original",
            "-DateTimeOriginal=2024:01:15 14:30:00",
            "-CreateDate=2024:01:15 14:30:00",
            "-ModifyDate=2024:01:15 14:30:00",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )

    return output_path


def generate_all_metadata_image() -> Path:
    """Generate image with GPS, camera settings, and datetime."""
    output_path = TARGET_DIR / "with_all.jpg"
    create_blank_image(output_path)

    _ = subprocess.run(
        [
            "exiftool",
            "-overwrite_original",
            # GPS
            "-GPSLatitude=40.7128",
            "-GPSLatitudeRef=N",
            "-GPSLongitude=74.0060",
            "-GPSLongitudeRef=W",
            # Camera
            "-Make=Sony",
            "-Model=Sony A7IV",
            "-ISO=800",
            "-FNumber=1.4",
            "-ExposureTime=1/200",
            # DateTime
            "-DateTimeOriginal=2024:06:20 16:45:30",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )

    return output_path


def generate_no_exif_image() -> Path:
    """Generate image stripped of all metadata."""
    output_path = TARGET_DIR / "no_exif.jpg"
    create_blank_image(output_path)

    # Strip all EXIF data
    _ = subprocess.run(
        [
            "exiftool",
            "-all=",
            "-overwrite_original",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )

    return output_path


def update_manifest(generated_files: list[Path]) -> None:
    """
    Update MANIFEST.md5 with generated EXIF test images.

    Args:
        generated_files: List of generated image file paths
    """
    # Read existing manifest
    existing_entries: list[str] = []
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    existing_entries.append(line)

    # Add new entries
    new_entries: list[str] = []
    for file_path in generated_files:
        # Use logical prefix 'test_media/' for manifest regardless of physical location
        path_str = f"test_media/exif_generated/{file_path.name}"
        md5_hash = calculate_md5(file_path)
        new_entries.append(f"{md5_hash}  {path_str}")

    # Write updated manifest
    from datetime import datetime

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        _ = f.write("# Test Media Manifest\n")
        _ = f.write(f"# Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        _ = f.write(f"# Source: {os.getenv('TEST_MEDIA_SOURCE', str(Path.home() / 'test_media'))}\n")
        _ = f.write("\n")

        for entry in existing_entries:
            _ = f.write(entry + "\n")

        if new_entries:
            _ = f.write("\n# Generated EXIF test images\n")
            for entry in new_entries:
                _ = f.write(entry + "\n")



def main() -> None:
    """Main execution function."""

    # Check exiftool
    if not check_exiftool_installed():
        return

    # Create output directory
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Generate test images
    generated_files: list[Path] = []

    generated_files.append(generate_gps_image())
    generated_files.append(generate_camera_settings_image())
    generated_files.append(generate_datetime_image())
    generated_files.append(generate_all_metadata_image())
    generated_files.append(generate_no_exif_image())


    # Update manifest
    update_manifest(generated_files)

    for _file_path in generated_files:
        pass


if __name__ == "__main__":
    main()
