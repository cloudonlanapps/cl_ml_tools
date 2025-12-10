import os

import subprocess
from werkzeug.exceptions import InternalServerError, NotFound
from typing import List


class FFMPEGCommands:
    def __init__(self):
        # self.video_bitrates = {"720": "3500k", "480": "1690k", "240": "326k"}
        self.video_bitrates = {"720": "3500k"}

    def toHLS(self, input_file: str, output_dir: str):
        if not os.path.exists(input_file):
            raise NotFound("input file doesn't exists")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Constructing filter complex part
        filter_complex = (
            f"[0:v]split={len(self.video_bitrates)}"
            + "".join([f"[{res}_in]" for res in self.video_bitrates.keys()])
            + ";"
            + ";".join(
                [
                    f"[{res}_in]scale=-2:{res}[{res}_out]"
                    for res in self.video_bitrates.keys()
                ]
            )
        )

        # Video and audio map commands
        video_map_commands = [
            item
            for res in self.video_bitrates.keys()
            for item in ["-map", f"[{res}_out]"]
        ]
        audio_map_commands = [
            item for _ in self.video_bitrates.keys() for item in ["-map", "0:a"]
        ]

        # Bitrate settings
        video_bitrate_commands = [
            item
            for i, res in enumerate(self.video_bitrates.keys())
            for item in [
                f"-b:v:{i}",
                self.video_bitrates[res],
                f"-maxrate:v:{i}",
                self.video_bitrates[res],
                f"-bufsize:v:{i}",
                self.video_bitrates[res],
            ]
        ]

        audio_bitrate_commands = [
            item
            for i in range(len(self.video_bitrates))
            for item in [f"-b:a:{i}", "128k"]
        ]

        # Constructing var_stream_map
        var_stream_map = " ".join(
            f"v:{i},a:{i},name:{res}p-{self.video_bitrates[res]}"
            for i, res in enumerate(self.video_bitrates.keys())
        )

        # Build command as a list instead of a string
        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_file,
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
            f"{output_dir}/adaptive-%v-%03d.ts",
            "-master_pl_name",
            "adaptive.m3u8",
            f"{output_dir}/adaptive-%v.m3u8",
        ]
        print(" ".join(command))

        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                raise InternalServerError(
                    "\n".join(["FFmpeg command failed", stderr, " ".join(command)])
                )

            if not os.path.exists(f"{output_dir}/adaptive.m3u8"):
                raise NotFound(
                    description="\n".join(
                        [
                            f"failed to open {output_dir}/adaptive.m3u8",
                            stderr,
                            " ".join(command),
                        ]
                    )
                )

        except Exception as e:
            raise InternalServerError(str(e))


if __name__ == "__main__":
    FFMPEGCommands().toHLS(
        input_file="/disks/data/git/github/asarangaram/dash_experiment/VID_20240206_095544.mp4",
        output_dir="/disks/data/git/github/asarangaram/dash_experiment/VID_20240206_095544",
    )
