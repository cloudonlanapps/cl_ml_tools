import subprocess
import math
import time


def get_video_properties(input_file):
    """Use FFprobe to get video properties."""
    ffprobe_command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        input_file,
    ]

    result = subprocess.run(ffprobe_command, capture_output=True, text=True)
    output = result.stdout.strip()

    if result.returncode != 0 or not output:
        raise RuntimeError(f"FFprobe failed: {result.stderr}")

    frame_rate_fraction, duration_str = output.split("\n")
    frame_rate_num, frame_rate_den = map(float, frame_rate_fraction.split("/"))
    duration = float(duration_str)

    fps = frame_rate_num / frame_rate_den
    frame_count = fps * duration

    return frame_count


def compute_tile_size(frame_count):
    """Determine tile size based on frame count."""
    if frame_count >= 16:
        return 4, 4  # 4x4 grid
    elif frame_count >= 9:
        return 3, 3  # 3x3 grid
    else:
        return 2, 2  # 2x2 grid


def create_video_thumbnail(input_file, output_file, dimension=256):
    """Generate a video thumbnail with a tiled grid."""
    # Step 1: Get video properties
    frame_count = get_video_properties(input_file)

    # Step 2: Compute tile size and frame frequency
    tile_size = compute_tile_size(frame_count)
    num_tiles = tile_size[0] * tile_size[1]
    frame_freq = max(
        1, (frame_count / num_tiles) // 1
    )  # Ensure frame_freq is at least 1

    # Step 3: Build and run the FFmpeg command
    ffmpeg_command = [
        "ffmpeg",
        "-loglevel",
        "panic",
        "-y",
        "-i",
        input_file,
        "-frames",
        "1",
        "-q:v",
        "1",
        "-vf",
        f"select=not(mod(n\\,{int(frame_freq)})),scale=-1:{dimension},tile={tile_size[0]}x{tile_size[1]}",
        output_file,
    ]

    subprocess.run(ffmpeg_command, check=True)
    print(f"Thumbnail created: {output_file}")


def create_video_thumbnail4x4(input_file, output_file, dimension=256):
    tile_size = (4, 4)
    # Step 3: Build and run the FFmpeg command
    ffmpeg_command = [
        "ffmpeg",
        "-loglevel",
        "panic",
        "-y",
        "-skip_frame",
        "nokey",
        "-i",
        input_file,
        "-frames",
        "1",
        "-q:v",
        "1",
        #'-vf', f'select=not(mod(n\\,{int(frame_freq)})),tile={tile_size[0]}x{tile_size[1]},scale=-1:{dimension}',
        "-vf",
        f"tile={tile_size[0]}x{tile_size[1]},loop={tile_size[0]*tile_size[1]}:1,scale=-1:{dimension}",
        output_file,
    ]

    subprocess.run(ffmpeg_command, check=True)
    print(f"Thumbnail created: {output_file}")


if __name__ == "__main__":
    # Example usage:
    video_file = (
        "/disks/backup/nalini_anand/oldPhone/Camera Roll/VID_20240216_095109.mp4"
    )
    output_thumbnail = "thumbnail_grid.jpg"

    start_time = time.time()
    create_video_thumbnail4x4(video_file, output_thumbnail)
    end_time = time.time()
    process_time = end_time - start_time
    print(f"process_time {process_time}")
