#!/usr/bin/env python3
"""Setup test media for cl_ml_tools tests.

This script:
1. Copies test media files from source directory to tests/test_media/
2. Generates MD5 checksums for all files
3. Creates MANIFEST.md5 file (outside test_media, in tests/ directory)
4. Creates .keep file in test_media/ directory for git
"""

import hashlib
import shutil
from pathlib import Path

import os
# Configuration
SOURCE_DIR = Path("/Users/anandasarangaram/Work/test_media")
TESTS_DIR = Path(__file__).parent
TARGET_DIR = Path(
    os.getenv("TEST_VECTORS_DIR", "/Users/anandasarangaram/Work/cl_server_test_media")
)
MANIFEST_FILE = TESTS_DIR / "MANIFEST.md5"


def calculate_md5(file_path: Path) -> str:
    """Calculate MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def copy_test_media() -> list[tuple[Path, str]]:
    """
    Copy test media from source to target directory.

    Returns:
        List of tuples (relative_path, md5_hash) for manifest
    """
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Source directory not found: {SOURCE_DIR}\n"
            f"Please ensure test media is available at this location.",
        )

    # Create target directory structure
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    (TARGET_DIR / "images").mkdir(exist_ok=True)
    (TARGET_DIR / "videos").mkdir(exist_ok=True)
    (TARGET_DIR / "audio").mkdir(exist_ok=True)

    manifest_entries = []

    # Copy images
    if (SOURCE_DIR / "images").exists():
        for src_file in (SOURCE_DIR / "images").iterdir():
            if src_file.is_file() and not src_file.name.startswith("."):
                dst_file = TARGET_DIR / "images" / src_file.name
                shutil.copy2(src_file, dst_file)
                relative_path = dst_file.relative_to(TESTS_DIR)
                md5_hash = calculate_md5(dst_file)
                manifest_entries.append((relative_path, md5_hash))

    # Copy videos
    if (SOURCE_DIR / "videos").exists():
        for src_file in (SOURCE_DIR / "videos").iterdir():
            if src_file.is_file() and not src_file.name.startswith("."):
                dst_file = TARGET_DIR / "videos" / src_file.name
                shutil.copy2(src_file, dst_file)
                relative_path = dst_file.relative_to(TESTS_DIR)
                md5_hash = calculate_md5(dst_file)
                manifest_entries.append((relative_path, md5_hash))

    # Copy audio
    if (SOURCE_DIR / "audio").exists():
        for src_file in (SOURCE_DIR / "audio").iterdir():
            if src_file.is_file() and not src_file.name.startswith("."):
                dst_file = TARGET_DIR / "audio" / src_file.name
                shutil.copy2(src_file, dst_file)
                relative_path = dst_file.relative_to(TESTS_DIR)
                md5_hash = calculate_md5(dst_file)
                manifest_entries.append((relative_path, md5_hash))

    return manifest_entries


def create_keep_file() -> None:
    """Create .keep file in test_media directory for git."""
    keep_file = TARGET_DIR / ".keep"
    keep_file.write_text("# Placeholder file for git - test_media/ is not committed\n")


def generate_manifest(entries: list[tuple[Path, str]]) -> None:
    """
    Generate MD5 manifest file.

    Args:
        entries: List of (relative_path, md5_hash) tuples
    """
    from datetime import datetime

    # Sort entries by path for consistent ordering
    entries.sort(key=lambda x: str(x[0]))

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        f.write("# Test Media Manifest\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Source: {SOURCE_DIR}\n")
        f.write("\n")

        for relative_path, md5_hash in entries:
            # Use forward slashes for cross-platform compatibility
            path_str = str(relative_path).replace("\\", "/")
            f.write(f"{md5_hash}  {path_str}\n")



def main() -> None:
    """Main execution function."""


    manifest_entries = copy_test_media()

    create_keep_file()

    generate_manifest(manifest_entries)



if __name__ == "__main__":
    main()
