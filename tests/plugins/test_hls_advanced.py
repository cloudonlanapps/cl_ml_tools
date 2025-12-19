"""Advanced unit tests for HLS stream generator and validator.

Targets error handling, validation failures, and edge cases in HLSStreamGenerator and HLSValidator.
"""

from unittest.mock import MagicMock, patch

import pytest

from cl_ml_tools.plugins.hls_streaming.algo.hls_stream_generator import (
    HLSStreamGenerator,
    HLSVariant,
    InternalServerError,
)
from cl_ml_tools.plugins.hls_streaming.algo.hls_validator import HLSValidator, validate_hls_output

# ============================================================================
# HLSVariant Tests
# ============================================================================


def test_hls_variant_string_init():
    """Test HLSVariant initialization with string resolution/bitrate."""
    v = HLSVariant(resolution="720", bitrate="3500")
    assert v.resolution == 720
    assert v.bitrate == 3500


def test_hls_variant_eq():
    """Test HLSVariant equality with non-HLSVariant."""
    v = HLSVariant(720, 3500)
    assert v != "not a variant"


def test_hls_variant_to_playlist_error():
    """Test toPlaylist raises ValueError for original stream (None resolution/bitrate)."""
    v = HLSVariant()
    with pytest.raises(ValueError, match="Cannot create playlist for original stream"):
        v.toPlaylist()


def test_hls_variant_get_resolution_error():
    """Test get_stream_resolution handles ffprobe failures."""
    v = HLSVariant(720, 3500)
    mock_result = MagicMock()
    mock_result.stdout = "{}"  # Missing streams key
    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(ValueError, match="Could not retrieve resolution information"):
            v.get_stream_resolution("/tmp")


# ============================================================================
# HLSStreamGenerator Tests
# ============================================================================


def test_hls_generator_check_stream_exception():
    """Test check_stream handles exceptions (e.g. ffprobe missing)."""
    with patch("subprocess.run", side_effect=Exception("ffprobe failed")):
        # We need a dummy generator, but __init__ calls check_stream
        with patch(
            "cl_ml_tools.plugins.hls_streaming.algo.hls_stream_generator.HLSStreamGenerator.check_stream",
            side_effect=[True, True],
        ):
            with patch(
                "cl_ml_tools.plugins.hls_streaming.algo.hls_stream_generator.HLSStreamGenerator.scan"
            ):
                gen = HLSStreamGenerator("input.mp4", "/tmp/out")

        # Now test the actual method with exception
        with patch("subprocess.run", side_effect=Exception("fail")):
            assert gen.check_stream("v") is False


def test_hls_generator_update_failures():
    """Test update method failures."""
    # Setup generator with mocks to avoid __init__ failures
    with patch(
        "cl_ml_tools.plugins.hls_streaming.algo.hls_stream_generator.HLSStreamGenerator.check_stream",
        return_value=True,
    ):
        with patch(
            "cl_ml_tools.plugins.hls_streaming.algo.hls_stream_generator.HLSStreamGenerator.scan"
        ):
            gen = HLSStreamGenerator("input.mp4", "/tmp/out")

    # Simulate FFmpeg failing to create master playlist
    with patch.object(gen, "run_command"):
        # Mocking specifically in the module namespace
        with patch(
            "cl_ml_tools.plugins.hls_streaming.algo.hls_stream_generator.os.path.exists",
            return_value=False,
        ):
            with patch(
                "cl_ml_tools.plugins.hls_streaming.algo.hls_stream_generator.m3u8.load",
                return_value=MagicMock(),
            ):
                with patch("cl_ml_tools.plugins.hls_streaming.algo.hls_stream_generator.os.remove"):
                    with pytest.raises(InternalServerError, match="ffmpeg didn't create master_pl"):
                        gen.update([HLSVariant(720, 3500)])

    # Simulate empty master playlist
    with patch.object(gen, "run_command"):
        with patch(
            "cl_ml_tools.plugins.hls_streaming.algo.hls_stream_generator.os.path.exists",
            return_value=True,
        ):
            mock_pl = MagicMock()
            mock_pl.playlists = []
            with patch(
                "cl_ml_tools.plugins.hls_streaming.algo.hls_stream_generator.m3u8.load",
                return_value=mock_pl,
            ):
                with patch("cl_ml_tools.plugins.hls_streaming.algo.hls_stream_generator.os.remove"):
                    with pytest.raises(
                        InternalServerError, match="no stream found in the create master_pl"
                    ):
                        gen.update([HLSVariant(720, 3500)])


def test_hls_generator_add_variants_errors():
    """Test addVariants error paths."""
    with patch(
        "cl_ml_tools.plugins.hls_streaming.algo.hls_stream_generator.HLSStreamGenerator.check_stream",
        return_value=True,
    ):
        with patch(
            "cl_ml_tools.plugins.hls_streaming.algo.hls_stream_generator.HLSStreamGenerator.scan"
        ):
            gen = HLSStreamGenerator("input.mp4", "/tmp/out")

    # 1. HLSVariant() in requested
    with pytest.raises(InternalServerError, match="original should be generated using addOriginal"):
        gen.addVariants([HLSVariant()])

    # 2. Validation failure after generation
    with patch.object(gen, "create"), patch.object(HLSVariant, "check", return_value=False):
        with pytest.raises(InternalServerError, match="is either invalid or partial or corrupted"):
            gen.addVariants([HLSVariant(720, 3500)])


def test_hls_generator_add_original_validation_fail():
    """Test addOriginal validation failure."""
    with patch(
        "cl_ml_tools.plugins.hls_streaming.algo.hls_stream_generator.HLSStreamGenerator.check_stream",
        return_value=True,
    ):
        with patch(
            "cl_ml_tools.plugins.hls_streaming.algo.hls_stream_generator.HLSStreamGenerator.scan"
        ):
            gen = HLSStreamGenerator("input.mp4", "/tmp/out")

    with patch.object(gen, "createOriginal"):
        with patch.object(
            HLSVariant, "check", side_effect=[False, False]
        ):  # Fails before and after create
            with pytest.raises(
                InternalServerError, match="is either invalid or partial or corrupted"
            ):
                gen.addOriginal()


# ============================================================================
# HLSValidator Tests
# ============================================================================


def test_hls_validator_missing_master():
    """Test HLSValidator handles missing master playlist."""
    validator = HLSValidator("missing.m3u8")
    result = validator.validate()
    assert result.is_valid is False
    assert "Master playlist not found" in result.errors[0]


def test_hls_validator_missing_variant():
    """Test HLSValidator handles missing variant playlist."""
    with patch("os.path.exists", side_effect=lambda p: "master.m3u8" in p):
        mock_master = MagicMock()
        mock_master.playlists = [MagicMock(uri="variant.m3u8")]
        with patch("m3u8.load", return_value=mock_master):
            validator = HLSValidator("master.m3u8")
            result = validator.validate()
            assert result.is_valid is False
            assert "Variant playlist not found" in result.errors[0]


def test_hls_validator_missing_segments():
    """Test HLSValidator handles missing segments."""

    # Mock master and variant results
    def exists_mock(p):
        if "master.m3u8" in p or "variant.m3u8" in p:
            return True
        return False  # Segments missing

    with patch("os.path.exists", side_effect=exists_mock):
        mock_master = MagicMock()
        mock_master.playlists = [
            MagicMock(uri="variant.m3u8", stream_info=MagicMock(bandwidth=1000))
        ]

        mock_variant = MagicMock()
        mock_variant.segments = [MagicMock(uri="seg1.ts"), MagicMock(uri="seg2.ts")]

        with patch("m3u8.load", side_effect=[mock_master, mock_variant]):
            validator = HLSValidator("master.m3u8")
            result = validator.validate()
            assert result.is_valid is False
            assert "Missing segments in variant.m3u8" in result.errors[0]


def test_hls_validator_exception():
    """Test HLSValidator handles parsing exceptions."""
    with patch("os.path.exists", return_value=True):
        with patch("m3u8.load", side_effect=ValueError("Parse error")):
            validator = HLSValidator("master.m3u8")
            result = validator.validate()
            assert result.is_valid is False
            assert "Validation error: Parse error" in result.errors[0]


def test_validate_hls_output_failure():
    """Test validate_hls_output returns error string on failure."""
    with patch(
        "cl_ml_tools.plugins.hls_streaming.algo.hls_validator.HLSValidator.validate"
    ) as mock_val:
        mock_result = MagicMock()
        mock_result.is_valid = False
        mock_result.errors = ["Some error"]
        mock_result.missing_files = ["file1"]
        mock_result.segments_found = 0
        mock_result.total_segments = 10
        mock_val.return_value = mock_result

        err = validate_hls_output("master.m3u8")
        assert err is not None
        assert "HLS validation failed" in err
        assert "Some error" in err
