import json
import os
import random
import re
import string
import subprocess
from typing import Any, cast, override

import m3u8
from m3u8.model import Playlist, StreamInfo

# ─────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────


class NotFound(Exception):
    pass


class InternalServerError(Exception):
    pass


# ─────────────────────────────────────────────
# HLS Variant
# ─────────────────────────────────────────────


class HLSVariant:
    def __init__(
        self,
        resolution: int | str | None = None,
        bitrate: int | None = None,
    ) -> None:
        if bitrate is not None:
            if isinstance(bitrate, str):
                bitrate = int(bitrate)
        if resolution is not None:
            if isinstance(resolution, str):
                resolution = int(resolution)

        self.resolution: int | None = resolution
        self.bitrate: int | None = bitrate

        self.resolution_str: str = str(resolution) if resolution is not None else "orig"
        self.scale_str: str = f"scale=-2:{resolution}" if resolution is not None else "copy"
        self.bitrate_str: str | None = f"{bitrate}k" if bitrate is not None else None

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HLSVariant):
            return False
        return self.resolution == other.resolution and self.bitrate == other.bitrate

    def uri(self) -> str:
        if self.resolution is None:
            return "adaptive-orig.m3u8"
        return f"adaptive-{self.resolution}p-{self.bitrate}.m3u8"

    def check(self, dir: str) -> bool:
        variant_path = os.path.join(dir, self.uri())
        if not os.path.exists(variant_path):
            return False

        playlist = m3u8.load(variant_path)
        for segment in playlist.segments:
            if segment.uri:
                if not os.path.exists(os.path.join(dir, segment.uri)):
                    return False
        return True

    def toPlaylist(self) -> Playlist:
        """Convert this variant to a Playlist object."""
        if self.resolution is None or self.bitrate is None:
            raise ValueError("Cannot create playlist for original stream")

        stream_info = StreamInfo(
            bandwidth=self.bitrate * 1000,
            resolution=(self.resolution, int(self.resolution * 16 / 9)),
            codecs="avc1.4d401f,mp4a.40.2",
        )

        playlist = Playlist(
            uri=self.uri(),
            stream_info=stream_info.__dict__,
            media=cast(Any, []),
            base_uri="",
        )
        return playlist

    def get_stream_resolution(self, dir: str) -> tuple[int, int]:
        variant_path = os.path.join(dir, self.uri())

        command: list[str] = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            variant_path,
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        try:
            info = cast(dict[str, list[dict[str, str]]], json.loads(result.stdout))
            stream = info["streams"][0]
            return int(stream["width"]), int(stream["height"])
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise ValueError("Could not retrieve resolution information") from exc


# ─────────────────────────────────────────────
# Stream Generator
# ─────────────────────────────────────────────


class HLSStreamGenerator:
    def __init__(self, input_file: str, output_dir: str) -> None:
        self.input_file: str = input_file
        self.output_dir: str = output_dir
        self.variants: list[HLSVariant] = []
        self.scan()

    # ─────────────────────────────────────────────
    #
    # ─────────────────────────────────────────────

    def scan(self) -> None:
        if not os.path.exists(self.input_file):
            raise NotFound("Input file does not exist")

        os.makedirs(self.output_dir, exist_ok=True)

        self.master_pl_name: str = "adaptive.m3u8"
        self.master_pl_path: str = os.path.join(self.output_dir, self.master_pl_name)

        self.variants.clear()

        if not os.path.exists(self.master_pl_path):
            return

        master_playlist = m3u8.load(self.master_pl_path)
        for playlist in master_playlist.playlists:
            match = None
            if playlist.uri:
                match = re.search(r"(\d+)p-(\d+)", playlist.uri)
            if match:
                variant = HLSVariant(
                    resolution=int(match.group(1)),
                    bitrate=int(match.group(2)),
                )
                if variant.check(self.output_dir):
                    self.variants.append(variant)

    # ─────────────────────────────────────────────
    #
    # ─────────────────────────────────────────────

    def getVariants(self) -> list[HLSVariant]:
        return self.variants

    def create(self, requested_variants: list[HLSVariant]):
        command = self.get_ffmpeg_command(
            requested_variants=requested_variants, master_pl_name="adaptive.m3u8"
        )
        self.run_command(command)

    def update(self, requested_variants: list[HLSVariant]):
        temp_master_pl_name = f"{''.join(random.choices(string.ascii_letters, k=10))}.m3u8"
        command = self.get_ffmpeg_command(
            requested_variants=requested_variants, master_pl_name=temp_master_pl_name
        )
        self.run_command(command)
        # merge master playlist
        path = os.path.join(self.output_dir, temp_master_pl_name)
        try:
            if os.path.exists(path):
                temp_playlist = m3u8.load(path)
                new_streams = temp_playlist.playlists
                if len(new_streams) > 0:
                    master_playlist = m3u8.load(self.master_pl_path)
                    for stream in new_streams:
                        master_playlist.playlists.append(stream)
                    master_playlist.playlists.sort(
                        key=lambda x: int(x.uri.split("-")[1].replace("p", "") if x.uri else 0),
                        reverse=True,
                    )
                    with open(self.master_pl_path, "w") as f:
                        _ = f.write(master_playlist.dumps())
                else:
                    raise InternalServerError(
                        f"no stream found in the create master_pl; {temp_master_pl_name}"
                    )
            else:
                raise InternalServerError(f"ffmpeg didn't create master_pl; {temp_master_pl_name}")
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            raise e
        finally:
            if os.path.exists(path):
                os.remove(path)

    def get_ffmpeg_command(self, requested_variants: list[HLSVariant], master_pl_name: str):
        # Constructing filter complex part
        split: list[str] = []
        scale: list[str] = []
        video_map_commands: list[str] = []
        audio_map_commands: list[str] = []
        video_bitrate_commands: list[str] = []
        audio_bitrate_commands: list[str] = []
        out_streams: list[str] = []
        for i, variant in enumerate(requested_variants):
            split.append(f"[{variant.resolution_str}_in]")
            scale.append(
                f"[{variant.resolution_str}_in]{variant.scale_str}[{variant.resolution_str}_out]"
            )
            video_map_commands.append("-map")
            video_map_commands.append(f"[{variant.resolution_str}_out]")
            audio_map_commands.append("-map")
            audio_map_commands.append("0:a")
            video_bitrate_commands.append(f"-b:v:{i}")
            if variant.bitrate_str:
                video_bitrate_commands.append(variant.bitrate_str)
            video_bitrate_commands.append(f"-maxrate:v:{i}")
            if variant.bitrate_str:
                video_bitrate_commands.append(variant.bitrate_str)
            video_bitrate_commands.append(f"-bufsize:v:{i}")
            if variant.bitrate_str:
                video_bitrate_commands.append(variant.bitrate_str)
            audio_bitrate_commands.append(f"-b:a:{i}")
            audio_bitrate_commands.append("128k")
            out_streams.append(f"v:{i},a:{i},name:{variant.resolution}p-{variant.bitrate}")

        filter_complex = (
            f"[0:v]split={len(requested_variants)}" + "".join(split) + ";" + ";".join(scale)
        )
        var_stream_map = " ".join(out_streams)

        master_pl_option = ["-master_pl_name", master_pl_name]

        command = [
            "ffmpeg",
            "-y",
            "-i",
            self.input_file,
            "-filter_complex",
            filter_complex,
            *video_map_commands,
            *audio_map_commands,
            *video_bitrate_commands,
            *audio_bitrate_commands,
            "-x264-params",
            "keyint=60:min-keyint=60:scenecut=0",
            "-var_stream_map",
            var_stream_map,
            "-hls_list_size",
            "0",
            "-hls_time",
            "2",
            "-hls_segment_filename",
            f"{self.output_dir}/adaptive-%v-%03d.ts",
            *master_pl_option,
            f"{self.output_dir}/adaptive-%v.m3u8",
        ]
        return command

    # ─────────────────────────────────────────────
    # FFmpeg helpers
    # ─────────────────────────────────────────────

    def run_command(self, command: list[str]) -> None:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise InternalServerError(
                "\n".join(["FFmpeg command failed", result.stderr, " ".join(command)])
            )

    def addVariants(self, requested_variants: list[HLSVariant]):
        print("addVariants")
        print(
            f"\tRequest to add {len(requested_variants)} variants. {','.join([item.uri() for item in requested_variants])}"
        )
        if HLSVariant() in requested_variants:
            raise InternalServerError("orignal should be generated using addOriginal")

        if len(self.variants) == 0:
            if len(requested_variants) > 0:
                print(
                    f"\tcreating stream with {len(requested_variants)} variants. {','.join([item.uri() for item in requested_variants])}"
                )
            self.create(requested_variants=requested_variants)
        else:
            print(
                f"\tStream with {len(self.variants)} variants found. {','.join([item.uri() for item in self.variants])}"
            )
            available_variants = self.variants
            found_variants = [item for item in requested_variants if item in available_variants]
            if len(found_variants) > 0:
                print(
                    f"\t{len(found_variants)} variant(s) is/are already present. {','.join([item.uri() for item in found_variants])}"
                )
            missing_variants = [
                item for item in requested_variants if item not in available_variants
            ]
            if len(missing_variants) > 0:
                if len(missing_variants) > 0:
                    print(
                        f"updating stream with {len(missing_variants)} variants. {','.join([item.uri() for item in missing_variants])}"
                    )
                self.update(missing_variants)

        # validate generated stream - may not be required if we check when creating HLSStreamGenerator
        for variant in requested_variants:
            valid = variant.check(dir=self.output_dir)
            if not valid:
                raise InternalServerError(
                    f"the stream generated {variant.uri()} is either invalid or partial or corrupted"
                )
        # reload
        self.scan()
        available_variants = self.variants
        missing_variants = [item for item in requested_variants if item not in available_variants]

        return len(missing_variants) == 0

    def createOriginal(self):
        command = [
            "ffmpeg",
            "-y",
            "-i",
            self.input_file,
            "-c",
            "copy",
            "-f",
            "hls",
            "-hls_time",
            "2",
            "-hls_segment_filename",
            f"{self.output_dir}/adaptive-orig-%03d.ts",
            f"{self.output_dir}/adaptive-orig.m3u8",
        ]
        print(" ".join(command))
        self.run_command(command)

        pass

    def addOriginal(self):
        print("addOriginal")
        print("\tReqest to convert the original stream to hls format without reencoding")
        # check if original is present
        variant = HLSVariant()
        valid = variant.check(dir=self.output_dir)
        if not valid:
            self.createOriginal()
            valid = variant.check(dir=self.output_dir)
            if not valid:
                raise InternalServerError(
                    f"the stream generated {variant.uri()} is either invalid or partial or corrupted"
                )
            return True
        print(f"\toriginal stream in hls format is already present. {variant.uri()}")
        return True


# ─────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    generator = HLSStreamGenerator(
        input_file="/disks/data/git/github/asarangaram/dash_experiment/VID-20230308-WA0129.mp4",
        output_dir="/disks/data/git/github/asarangaram/dash_experiment/random_folder",
    )

    res = generator.addVariants([HLSVariant(resolution=720, bitrate=900)])
    if not res:
        print("failed")
    res = generator.addVariants([HLSVariant(resolution=480, bitrate=400)])
    if not res:
        print("failed")

    res = generator.addVariants([HLSVariant(resolution=240, bitrate=200)])
    if not res:
        print("failed")

    res = generator.addOriginal()
    if not res:
        print("failed")

    print("done")
