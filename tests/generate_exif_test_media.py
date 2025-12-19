#!/usr/bin/env python3
"""Generate test images with known EXIF metadata using exiftool.

This script creates test images with specific EXIF metadata for testing
the EXIF extraction plugin. Requires exiftool to be installed.
"""

import hashlib
import subprocess
from pathlib import Path

from PIL import Image

# Configuration
TESTS_DIR = Path(__file__).parent
TARGET_DIR = TESTS_DIR / "test_media" / "exif_generated"
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
        if result.returncode == 0:
            print(f"✓ ExifTool version: {result.stdout.strip()}")
            return True
        return False
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

    subprocess.run(
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

    print(f"  ✓ Created: {output_path.name}")
    print("    GPS: 37.7749° N, 122.4194° W")
    return output_path


def generate_camera_settings_image() -> Path:
    """
    Generate image with camera settings.

    Settings: ISO 400, Aperture f/2.8, Shutter 1/100
    """
    output_path = TARGET_DIR / "with_camera.jpg"
    create_blank_image(output_path)

    subprocess.run(
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

    print(f"  ✓ Created: {output_path.name}")
    print("    Camera: Canon EOS R5, ISO 400, f/2.8, 1/100s")
    return output_path


def generate_datetime_image() -> Path:
    """
    Generate image with specific datetime.

    DateTime: 2024-01-15 14:30:00
    """
    output_path = TARGET_DIR / "with_datetime.jpg"
    create_blank_image(output_path)

    subprocess.run(
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

    print(f"  ✓ Created: {output_path.name}")
    print("    DateTime: 2024-01-15 14:30:00")
    return output_path


def generate_all_metadata_image() -> Path:
    """Generate image with GPS, camera settings, and datetime."""
    output_path = TARGET_DIR / "with_all.jpg"
    create_blank_image(output_path)

    subprocess.run(
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

    print(f"  ✓ Created: {output_path.name}")
    print("    GPS + Camera + DateTime (New York)")
    return output_path


def generate_no_exif_image() -> Path:
    """Generate image stripped of all metadata."""
    output_path = TARGET_DIR / "no_exif.jpg"
    create_blank_image(output_path)

    # Strip all EXIF data
    subprocess.run(
        [
            "exiftool",
            "-all=",
            "-overwrite_original",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )

    print(f"  ✓ Created: {output_path.name}")
    print("    No EXIF metadata")
    return output_path


def update_manifest(generated_files: list[Path]) -> None:
    """
    Update MANIFEST.md5 with generated EXIF test images.

    Args:
        generated_files: List of generated image file paths
    """
    # Read existing manifest
    existing_entries = []
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    existing_entries.append(line)

    # Add new entries
    new_entries = []
    for file_path in generated_files:
        relative_path = file_path.relative_to(TESTS_DIR)
        md5_hash = calculate_md5(file_path)
        path_str = str(relative_path).replace("\\", "/")
        new_entries.append(f"{md5_hash}  {path_str}")

    # Write updated manifest
    from datetime import datetime

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        f.write("# Test Media Manifest\n")
        f.write(f"# Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# Source: /Users/anandasarangaram/Work/test_media\n")
        f.write("\n")

        for entry in existing_entries:
            f.write(entry + "\n")

        if new_entries:
            f.write("\n# Generated EXIF test images\n")
            for entry in new_entries:
                f.write(entry + "\n")

    print(f"\n✓ Updated manifest: {MANIFEST_FILE.relative_to(TESTS_DIR)}")
    print(f"  Added {len(new_entries)} EXIF test images")


def main() -> None:
    """Main execution function."""
    print("=" * 70)
    print("Generating EXIF test media for cl_ml_tools")
    print("=" * 70)
    print()

    # Check exiftool
    print("Step 1: Checking for exiftool...")
    if not check_exiftool_installed():
        print("\n❌ Error: exiftool not found!")
        print("\nPlease install exiftool:")
        print("  macOS:  brew install exiftool")
        print("  Linux:  apt-get install libimage-exiftool-perl")
        return
    print()

    # Create output directory
    print("Step 2: Creating output directory...")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Created: {TARGET_DIR.relative_to(TESTS_DIR)}")
    print()

    # Generate test images
    print("Step 3: Generating test images with EXIF metadata...")
    generated_files = []

    generated_files.append(generate_gps_image())
    generated_files.append(generate_camera_settings_image())
    generated_files.append(generate_datetime_image())
    generated_files.append(generate_all_metadata_image())
    generated_files.append(generate_no_exif_image())

    print(f"\n  ✓ Generated {len(generated_files)} test images")
    print()

    # Update manifest
    print("Step 4: Updating MANIFEST.md5...")
    update_manifest(generated_files)
    print()

    print("=" * 70)
    print("✓ EXIF test media generation complete!")
    print("=" * 70)
    print()
    print("Generated files:")
    for file_path in generated_files:
        print(f"  - {file_path.relative_to(TESTS_DIR)}")
    print()
    print("These files can now be used in EXIF extraction tests with known metadata.")


if __name__ == "__main__":
    main()
