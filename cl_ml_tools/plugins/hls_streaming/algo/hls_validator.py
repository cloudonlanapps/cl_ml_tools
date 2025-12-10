import os
import re
import subprocess
from typing import Dict, List, Tuple
from dataclasses import dataclass
import m3u8


@dataclass
class ValidationResult:
    is_valid: bool
    missing_files: List[str]
    total_segments: int
    segments_found: int
    variants_info: Dict[str, Dict]
    errors: List[str]


class HLSValidator:
    def __init__(self, m3u8_file: str):
        self.m3u8_file = m3u8_file
        self.output_dir = os.path.dirname(m3u8_file)
        self.errors = []

    def validate(self) -> ValidationResult:
        """Validate the HLS output including master playlist and all variant playlists."""
        missing_files = []
        variants_info = {}
        total_segments = 0
        segments_found = 0

        try:
            # First check if master playlist exists
            master_path = self.m3u8_file
            if not os.path.exists(master_path):
                self.errors.append(f"Master playlist not found: {master_path}")
                return ValidationResult(
                    is_valid=False,
                    missing_files=[master_path],
                    total_segments=0,
                    segments_found=0,
                    variants_info={},
                    errors=self.errors,
                )

            # Parse master playlist
            master_playlist = m3u8.load(master_path)

            # Check each variant stream
            for playlist in master_playlist.playlists:
                uri = playlist.uri
                variant_path = os.path.join(self.output_dir, uri)

                if not os.path.exists(variant_path):
                    missing_files.append(variant_path)
                    self.errors.append(f"Variant playlist not found: {variant_path}")
                    continue

                # Parse variant playlist
                variant = m3u8.load(variant_path)

                # Extract resolution from URI (assuming format like adaptive-720p-3500k.m3u8)
                resolution_match = re.search(r"(\d+)p-(\d+)k", uri)
                if resolution_match:
                    resolution = resolution_match.group(1)
                else:
                    resolution = "unknown"

                # Check segments
                variant_segments = variant.segments
                total_segments += len(variant_segments)
                segments_present = 0
                missing_segments = []

                for segment in variant_segments:
                    segment_path = os.path.join(self.output_dir, segment.uri)
                    if os.path.exists(segment_path):
                        segments_present += 1
                    else:
                        missing_segments.append(segment.uri)
                        missing_files.append(segment_path)

                segments_found += segments_present

                # Store variant information
                variants_info[uri] = {
                    "resolution": resolution,
                    "bandwidth": playlist.stream_info.bandwidth,
                    "total_segments": len(variant_segments),
                    "segments_present": segments_present,
                    "missing_segments": missing_segments,
                }

                # Check if all segments are present
                if segments_present != len(variant_segments):
                    self.errors.append(
                        f"Missing segments in {uri}: {len(missing_segments)} of {len(variant_segments)}"
                    )

        except Exception as e:
            self.errors.append(f"Validation error: {str(e)}")
            return ValidationResult(
                is_valid=False,
                missing_files=missing_files,
                total_segments=total_segments,
                segments_found=segments_found,
                variants_info=variants_info,
                errors=self.errors,
            )

        is_valid = len(missing_files) == 0 and len(self.errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            missing_files=missing_files,
            total_segments=total_segments,
            segments_found=segments_found,
            variants_info=variants_info,
            errors=self.errors,
        )


def validate_hls_output(m3u8_file: str) -> ValidationResult:
    validator = HLSValidator(m3u8_file=m3u8_file)
    validation_result = validator.validate()
    print(validation_result)
    if not validation_result.is_valid:
        error_message = "\n".join(
            [
                "HLS validation failed:",
                *validation_result.errors,
                f"Missing files: {len(validation_result.missing_files)}",
                f"Segments found: {validation_result.segments_found}/{validation_result.total_segments}",
            ]
        )
        return error_message
    return None


if __name__ == "__main__":
    result = validate_hls_output(
        m3u8_file="/disks/data/git/github/asarangaram/dash_experiment/VID_20240206_095544/adaptive.m3u8"
    )
    print(result)
