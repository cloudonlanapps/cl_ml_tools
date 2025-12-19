"""Model download and caching utility for ONNX models.

This module provides utilities to download ONNX models from external sources
(Hugging Face, ONNX Model Zoo, etc.) and cache them locally for reuse.

IMPORTANT: This module does not redistribute models. It only downloads models
from their original sources and caches them locally. Users must comply with
the original model licenses.
"""

import hashlib
import logging
from pathlib import Path
from typing import cast

import httpx

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Download and cache ONNX models locally.

    Models are cached in ~/.cache/cl_ml_tools/models/ by default.
    """

    def __init__(self, cache_dir: Path | None = None):
        """Initialize the model downloader.

        Args:
            cache_dir: Directory to cache models. Defaults to ~/.cache/cl_ml_tools/models/
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "cl_ml_tools" / "models"

        self.cache_dir: Path = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(
        self,
        url: str,
        filename: str,
        expected_sha256: str | None = None,
        force_redownload: bool = False,
        auto_extract: bool = False,
        extract_pattern: str | None = None,
    ) -> Path:
        """Download a model file and optionally extract from ZIP archive.

        Args:
            url: URL to download the model from
            filename: Filename to save the model as (e.g., "face_detection.onnx" or "model.zip")
            expected_sha256: Expected SHA256 hash for verification (optional but recommended)
            force_redownload: If True, redownload even if file exists
            auto_extract: If True and file is .zip, extract contents
            extract_pattern: Glob pattern to find file in ZIP (e.g., "*.onnx").
                           If None, extracts all files and returns first extracted file.

        Returns:
            Path to the cached model file (or extracted file if auto_extract=True)

        Raises:
            httpx.HTTPError: If download fails
            ValueError: If SHA256 hash does not match expected value
            FileNotFoundError: If extract_pattern specified but no matching files found in ZIP
        """
        model_path = self.cache_dir / filename
        file_downloaded = False

        # Check if model already exists and is valid
        if model_path.exists() and not force_redownload:
            if expected_sha256 is None:
                logger.info(f"Model already cached: {model_path}")
                file_downloaded = True
            else:
                # Verify existing file hash
                actual_hash = self._compute_sha256(model_path)
                if actual_hash == expected_sha256:
                    logger.info(f"Model already cached and verified: {model_path}")
                    file_downloaded = True
                else:
                    logger.warning(
                        f"Cached model hash mismatch. Expected {expected_sha256}, "
                        + f"got {actual_hash}. Re-downloading..."
                    )

        if not file_downloaded:
            # Download the model
            logger.info(f"Downloading model from {url} to {model_path}")

            try:
                with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as response:
                    _ = response.raise_for_status()

                    # Get total size if available
                    size_header = cast(str | None, response.headers.get("content-length"))
                    total_size: int = int(size_header) if size_header is not None else 0

                    # Download with progress logging
                    with open(model_path, "wb") as f:
                        downloaded = 0
                        for chunk in response.iter_bytes(chunk_size=8192):
                            _ = f.write(chunk)
                            downloaded += len(chunk)

                            # Log progress every 10MB
                            if downloaded % (10 * 1024 * 1024) == 0 and total_size > 0:
                                progress = (downloaded / total_size) * 100
                                logger.info(f"Download progress: {progress:.1f}%")

                logger.info(f"Download complete: {model_path}")

                # Verify hash if provided
                if expected_sha256 is not None:
                    actual_hash = self._compute_sha256(model_path)
                    if actual_hash != expected_sha256:
                        model_path.unlink()  # Delete corrupted file
                        raise ValueError(
                            f"Downloaded model hash mismatch. Expected {expected_sha256}, "
                            + f"got {actual_hash}. File deleted."
                        )
                    logger.info("Model hash verified successfully")

            except httpx.HTTPError as e:
                logger.error(f"Failed to download model from {url}: {e}")
                raise

        # Auto-extract if requested
        if auto_extract and model_path.suffix == ".zip":
            import zipfile

            extract_dir = model_path.parent

            # Check if already extracted
            if extract_pattern:
                # Note: This simple glob might not work recursively if pattern doesn't contain **
                # But it matches how we look for it below.
                # However, for recursive search inside zip vs flat glob here, there might be mismatch.
                # Let's trust usage of extract_pattern.
                # If checking existing files, we need to be careful.
                # If we assume extraction happened, we can check.
                # But if previous run didn't extract (because of the bug), we must extract now.

                # If we can't easily check for *all* files, maybe best to just try extracting.
                # Optimisation: Check for at least one match?
                # Using rglob if pattern indicates recursion?
                existing = list(extract_dir.rglob(extract_pattern)) if "**" in extract_pattern else list(extract_dir.glob(extract_pattern))
                if existing and not force_redownload:
                     logger.info(f"Found already extracted file: {existing[0]}")
                     return existing[0]

            # Extract ZIP file
            logger.info(f"Extracting {model_path}")
            with zipfile.ZipFile(model_path, "r") as zip_ref:
                if extract_pattern:
                    # Extract specific files matching pattern
                    # Using Path(m).match(extract_pattern) supports basic glob
                    members = [
                        m for m in zip_ref.namelist() if Path(m).match(extract_pattern)
                    ]
                    if not members:
                        raise FileNotFoundError(
                            f"No files matching '{extract_pattern}' found in {model_path}"
                        )
                    for member in members:
                        _ = zip_ref.extract(member, extract_dir)

                    # Return path to the first matched file, resolving subdirectories
                    extracted_file = extract_dir / members[0]
                else:
                    # Extract all files
                    zip_ref.extractall(extract_dir)
                    extracted_file = extract_dir / zip_ref.namelist()[0]

            logger.info(f"Extracted to: {extracted_file}")
            return extracted_file

        return model_path

    def _compute_sha256(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            Hexadecimal SHA256 hash string
        """
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def get_cached_model_path(self, filename: str) -> Path | None:
        """Get path to a cached model if it exists.

        Args:
            filename: Model filename (e.g., "face_detection.onnx")

        Returns:
            Path to cached model if it exists, None otherwise
        """
        model_path = self.cache_dir / filename
        return model_path if model_path.exists() else None

    def clear_cache(self) -> None:
        """Delete all cached models."""
        if self.cache_dir.exists():
            for model_file in self.cache_dir.glob("*.onnx"):
                model_file.unlink()
                logger.info(f"Deleted cached model: {model_file}")


# Singleton instance for shared use across plugins
_default_downloader: ModelDownloader | None = None


def get_model_downloader() -> ModelDownloader:
    """Get the default model downloader instance.

    Returns:
        ModelDownloader instance
    """
    global _default_downloader
    if _default_downloader is None:
        _default_downloader = ModelDownloader()
    return _default_downloader
