"""Unit tests for ModelDownloader."""

import hashlib
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from cl_ml_tools.utils.model_downloader import ModelDownloader, get_model_downloader


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Fixture for temporary cache directory."""
    d = tmp_path / "cache"
    d.mkdir()
    return d


@pytest.fixture
def downloader(cache_dir: Path) -> ModelDownloader:
    """Fixture for ModelDownloader instance."""
    return ModelDownloader(cache_dir=cache_dir)


def test_get_model_downloader():
    """Test get_model_downloader singleton."""
    d1 = get_model_downloader()
    d2 = get_model_downloader()
    assert d1 is d2
    assert isinstance(d1, ModelDownloader)


def test_init_defaults():
    """Test initialization with default cache directory."""
    with patch("pathlib.Path.home", return_value=Path("/home/user")):
        with patch("pathlib.Path.mkdir"):
            d = ModelDownloader()
            assert d.cache_dir == Path("/home/user/.cache/cl_ml_tools/models")


def test_compute_sha256(downloader: ModelDownloader, tmp_path: Path):
    """Test SHA256 computation."""
    test_file = tmp_path / "test.txt"
    content = b"hello world"
    _ = test_file.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    assert downloader._compute_sha256(test_file) == expected  # pyright: ignore[reportPrivateUsage]


def test_download_exists_no_hash(downloader: ModelDownloader, cache_dir: Path):
    """Test download when file exists and no hash provided."""
    filename = "model.onnx"
    model_path = cache_dir / filename
    _ = model_path.write_bytes(b"data")

    path = downloader.download(url="http://example.com/model.onnx", filename=filename)
    assert path == model_path
    # No httpx call should happen (we didn't mock it, so it would fail if it did)


def test_download_exists_with_valid_hash(downloader: ModelDownloader, cache_dir: Path):
    """Test download when file exists and hash matches."""
    filename = "model.onnx"
    model_path = cache_dir / filename
    content = b"data"
    _ = model_path.write_bytes(content)
    expected_hash = hashlib.sha256(content).hexdigest()

    path = downloader.download(
        url="http://example.com/model.onnx", filename=filename, expected_sha256=expected_hash
    )
    assert path == model_path


def test_download_exists_with_invalid_hash_redownloads(
    downloader: ModelDownloader, cache_dir: Path
):
    """Test download when file exists but hash mismatches."""
    filename = "model.onnx"
    model_path = cache_dir / filename
    _ = model_path.write_bytes(b"old_data")

    new_content = b"new_data"
    new_hash = hashlib.sha256(new_content).hexdigest()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-length": str(len(new_content))}
    mock_response.iter_bytes.return_value = [new_content]
    mock_response.raise_for_status.return_value = None

    with patch("httpx.stream") as mock_stream:
        mock_stream.return_value.__enter__.return_value = mock_response

        path = downloader.download(
            url="http://example.com/model.onnx", filename=filename, expected_sha256=new_hash
        )

        assert path == model_path
        assert model_path.read_bytes() == new_content


def test_download_http_error(downloader: ModelDownloader):
    """Test download failure due to HTTP error."""
    with patch("httpx.stream") as mock_stream:
        mock_enter = mock_stream.return_value.__enter__.return_value
        mock_enter.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=MagicMock(), response=MagicMock()
        )

        with pytest.raises(httpx.HTTPStatusError):
            _ = downloader.download(url="http://example.com/404", filename="fail.onnx")


def test_download_hash_mismatch_after_download(downloader: ModelDownloader, cache_dir: Path):
    """Test hash verification failure after download."""
    filename = "model.onnx"
    content = b"corrupted_data"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-length": str(len(content))}
    mock_response.iter_bytes.return_value = [content]
    mock_response.raise_for_status.return_value = None

    with patch("httpx.stream") as mock_stream:
        mock_stream.return_value.__enter__.return_value = mock_response

        with pytest.raises(ValueError, match="Downloaded model hash mismatch"):
            _ = downloader.download(
                url="http://example.com/model.onnx", filename=filename, expected_sha256="wrong_hash"
            )

    assert not (cache_dir / filename).exists()


def test_download_and_extract_zip(downloader: ModelDownloader, cache_dir: Path, tmp_path: Path):
    """Test downloading and extracting a ZIP file."""
    # Create a real ZIP for testing extraction
    zip_filename = "model.zip"
    zip_path = tmp_path / zip_filename
    onnx_filename = "model.onnx"
    onnx_content = b"onnx_data"

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(onnx_filename, onnx_content)
        zf.writestr("other.txt", b"other")

    zip_bytes = zip_path.read_bytes()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-length": str(len(zip_bytes))}
    mock_response.iter_bytes.return_value = [zip_bytes]
    mock_response.raise_for_status.return_value = None

    with patch("httpx.stream") as mock_stream:
        mock_stream.return_value.__enter__.return_value = mock_response

        # Test without pattern (should return first file)
        path = downloader.download(
            url="http://example.com/model.zip", filename=zip_filename, auto_extract=True
        )
        assert path.name == onnx_filename
        assert path.read_bytes() == onnx_content

        # Test with pattern
        path = downloader.download(
            url="http://example.com/model.zip",
            filename=zip_filename,
            auto_extract=True,
            extract_pattern="*.onnx",
        )
        assert path.name == onnx_filename
        assert path.suffix == ".onnx"


def test_get_cached_model_path(downloader: ModelDownloader, cache_dir: Path):
    """Test get_cached_model_path."""
    assert downloader.get_cached_model_path("missing.onnx") is None

    _ = (cache_dir / "exists.onnx").write_bytes(b"data")
    assert downloader.get_cached_model_path("exists.onnx") == cache_dir / "exists.onnx"


def test_clear_cache(downloader: ModelDownloader, cache_dir: Path):
    """Test clear_cache."""
    _ = (cache_dir / "model1.onnx").write_bytes(b"1")
    _ = (cache_dir / "model2.onnx").write_bytes(b"2")
    _ = (cache_dir / "other.txt").write_bytes(b"3")

    downloader.clear_cache()

    assert not (cache_dir / "model1.onnx").exists()
    assert not (cache_dir / "model2.onnx").exists()
    assert (cache_dir / "other.txt").exists()
