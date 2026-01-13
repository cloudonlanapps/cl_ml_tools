import os
import subprocess

from loguru import logger


class NotFound(Exception):
    """Raised when a required file or resource is missing."""


class InternalServerError(Exception):
    """Raised when ffmpeg execution fails."""


class FFMPEGCommands:
    def __init__(self) -> None:
        # Resolution (height) → bitrate
        self.video_bitrates: dict[str, str] = {
            "720": "3500k",
        }

    def to_hls(self, input_file: str, output_dir: str) -> None:
        if not os.path.exists(input_file):
            raise NotFound("Input file does not exist")

        if not os.path.exists(output_dir):
            raise NotFound("Output directory does not exist")

        # ─────────────────────────────────────────────
        # Filter complex
        # ─────────────────────────────────────────────
        filter_complex: str = (
            f"[0:v]split={len(self.video_bitrates)}"
            + "".join(f"[{res}_in]" for res in self.video_bitrates)
            + ";"
            + ";".join(
                f"[{res}_in]scale=-2:{res}[{res}_out]" for res in self.video_bitrates
            )
        )

        # ─────────────────────────────────────────────
        # Stream mapping
        # ─────────────────────────────────────────────
        video_map_commands: list[str] = [
            item for res in self.video_bitrates for item in ("-map", f"[{res}_out]")
        ]

        audio_map_commands: list[str] = [
            item for _ in self.video_bitrates for item in ("-map", "0:a")
        ]

        # ─────────────────────────────────────────────
        # Bitrate settings
        # ─────────────────────────────────────────────
        video_bitrate_commands: list[str] = [
            item
            for i, res in enumerate(self.video_bitrates)
            for item in (
                f"-b:v:{i}",
                self.video_bitrates[res],
                f"-maxrate:v:{i}",
                self.video_bitrates[res],
                f"-bufsize:v:{i}",
                self.video_bitrates[res],
            )
        ]

        audio_bitrate_commands: list[str] = [
            item
            for i in range(len(self.video_bitrates))
            for item in (f"-b:a:{i}", "128k")
        ]

        # ─────────────────────────────────────────────
        # Variant stream map
        # ─────────────────────────────────────────────
        var_stream_map: str = " ".join(
            f"v:{i},a:{i},name:{res}p-{self.video_bitrates[res]}"
            for i, res in enumerate(self.video_bitrates)
        )

        command: list[str] = [
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

        logger.debug(" ".join(command))

        try:
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise InternalServerError(f"Failed to start ffmpeg: {exc}") from exc

        if process.returncode != 0:
            raise InternalServerError(
                "\n".join(
                    [
                        "FFmpeg command failed",
                        process.stderr,
                        " ".join(command),
                    ]
                )
            )

        master_playlist: str = os.path.join(output_dir, "adaptive.m3u8")
        if not os.path.exists(master_playlist):
            raise NotFound(
                "\n".join(
                    [
                        f"Failed to create {master_playlist}",
                        process.stderr,
                        " ".join(command),
                    ]
                )
            )
