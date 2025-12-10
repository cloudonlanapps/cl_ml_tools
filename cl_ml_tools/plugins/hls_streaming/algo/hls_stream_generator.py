import json
import os
import random
import re
import string
import subprocess
from werkzeug.exceptions import InternalServerError, NotFound
from typing import List
import m3u8


## TODO:
## IMPLEMENT removeVariant


class HLSVariant:
    def __init__(self, resolution: int = None, bitrate: int = None):
        if bitrate is not None:
            if isinstance(bitrate, str):
                bitrate = int(bitrate)
            if not isinstance(bitrate, int):
                raise ValueError("bitrate must be int or int convertable string")
        if resolution is not None:
            if isinstance(resolution, str):
                resolution = int(resolution)
            if not isinstance(resolution, int):
                raise ValueError("resolution must be int or int convertable string")
        self.resolution = resolution
        self.bitrate = bitrate
        self.resolution_str = f"{resolution}" if resolution is not None else "orig"
        self.scale_str = f"scale=-2:{resolution}" if resolution is not None else "copy"
        self.bitrate_str = f"{bitrate}k" if resolution is not None else None
        self.dir = dir
        pass

    def __eq__(self, other):
        # Custom comparison: matching resolution and bitrate
        if isinstance(other, HLSVariant):
            return self.resolution == other.resolution and self.bitrate == other.bitrate
        return False

    def uri(self):
        if self.resolution is None:
            return f"adaptive-orig.m3u8"
        return f"adaptive-{self.resolution}p-{self.bitrate}.m3u8"

    def check(self, dir: str):
        # from URI (assuming format like adaptive-720p-3500k.m3u8)

        variant_path = os.path.join(dir, self.uri())
        if not os.path.exists(variant_path):
            return False

        variant_playlist = m3u8.load(variant_path)
        total_segments = len(variant_playlist.segments)

        for segment in variant_playlist.segments:
            segment_path = os.path.join(dir, segment.uri)
            if not os.path.exists(segment_path):
                return False
        return True

    def toPlayList(self):
        bitrate = self.bitrate
        resolution = self.resolution
        return m3u8.model.Playlist(
            uri=self.uri(),
            stream_info=m3u8.model.StreamInfo(
                bandwidth=int(bitrate.replace("k", "000")),
                resolution=f"{resolution}x{int(resolution * 16/9)}",  # Height will be incorrect.
                codecs="avc1.4d401f,mp4a.40.2",
                video="video",
            ),
        )

    def get_stream_resolution(self, dir: str):
        variant_path = os.path.join(dir, self.uri())
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            f"{variant_path}",
        ]

        # Execute command and capture output
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        output = result.stdout

        # Parse the JSON output
        try:
            info = json.loads(output)
            width = info["streams"][0]["width"]
            height = info["streams"][0]["height"]
            return width, height
        except (KeyError, IndexError, json.JSONDecodeError):
            raise ValueError("Could not retrieve resolution information")


class HLSStreamGenerator:
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = output_dir
        self.scan()

    def scan(self):
        if not os.path.exists(self.input_file):
            raise NotFound("input file doesn't exists")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.master_pl_name = "adaptive.m3u8"
        self.master_pl_path = os.path.join(self.output_dir, self.master_pl_name)
        self.master_pl_exists = os.path.exists(self.master_pl_path)

        self.variants = []
        if self.master_pl_exists:
            master_playlist = m3u8.load(self.master_pl_path)
            for playlist in master_playlist.playlists:
                uri = playlist.uri
                # Extract resolution from URI (assuming format like adaptive-720p-3500k.m3u8)
                match = re.search(r"(\d+)p-(\d+)", uri)
                if match:
                    resolution = match.group(1)
                    bitrate = match.group(2)
                    variant = HLSVariant(resolution=resolution, bitrate=bitrate)
                    if variant.check(dir=self.output_dir):
                        self.variants.append(variant)

    def getVariants(self):
        return self.variants

    def create(self, requested_variants: List[HLSVariant]):
        command = self.get_ffmpeg_command(
            requested_variants=requested_variants, master_pl_name="adaptive.m3u8"
        )
        self.run_command(command)

    def update(self, requested_variants: List[HLSVariant]):
        temp_master_pl_name = (
            f"{''.join(random.choices(string.ascii_letters, k=10))}.m3u8"
        )
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
                        key=lambda x: int(x.uri.split("-")[1].replace("p", "")),
                        reverse=True,
                    )
                    with open(self.master_pl_path, "w") as f:
                        f.write(master_playlist.dumps())
                else:
                    raise InternalServerError(
                        f"no stream found in the create master_pl; {temp_master_pl_name}"
                    )
            else:
                raise InternalServerError(
                    f"ffmpeg didn't create master_pl; {temp_master_pl_name}"
                )
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            raise e
        finally:
            if os.path.exists(path):
                os.remove(path)

    def get_ffmpeg_command(
        self, requested_variants: List[HLSVariant], master_pl_name: str
    ):
        # Constructing filter complex part
        split = []
        scale = []
        video_map_commands = []
        audio_map_commands = []
        video_bitrate_commands = []
        audio_bitrate_commands = []
        out_streams = []
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
            video_bitrate_commands.append(variant.bitrate_str)
            video_bitrate_commands.append(f"-maxrate:v:{i}")
            video_bitrate_commands.append(variant.bitrate_str)
            video_bitrate_commands.append(f"-bufsize:v:{i}")
            video_bitrate_commands.append(variant.bitrate_str)
            audio_bitrate_commands.append(f"-b:a:{i}")
            audio_bitrate_commands.append("128k")
            out_streams.append(
                f"v:{i},a:{i},name:{variant.resolution}p-{variant.bitrate}"
            )

        filter_complex = (
            f"[0:v]split={len(requested_variants)}"
            + "".join(split)
            + ";"
            + ";".join(scale)
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

    def run_command(self, command):
        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                raise InternalServerError(
                    "\n".join(["FFmpeg command failed", stderr, " ".join(command)])
                )

        except Exception as e:
            raise InternalServerError(str(e))

        return command

    def addVariants(self, requested_variants: List[HLSVariant]):
        print("addVariants")
        print(
            f"\tRequest to add {len(requested_variants)} variants. { ','.join([item.uri() for item in requested_variants])}"
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
                f"\tStream with {len(self.variants)} variants found. { ','.join([item.uri() for item in self.variants])}"
            )
            available_variants = self.variants
            found_variants = [
                item for item in requested_variants if item in available_variants
            ]
            if len(found_variants) > 0:
                print(
                    f"\t{len(found_variants)} variant(s) is/are already present. { ','.join([item.uri() for item in found_variants])}"
                )
            missing_variants = [
                item for item in requested_variants if item not in available_variants
            ]
            if len(missing_variants) > 0:
                if len(missing_variants) > 0:
                    print(
                        f"updating stream with {len(missing_variants)} variants. { ','.join([item.uri() for item in missing_variants])}"
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
        missing_variants = [
            item for item in requested_variants if item not in available_variants
        ]

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
        print(
            "\tReqest to convert the original stream to hls format without reencoding"
        )
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
