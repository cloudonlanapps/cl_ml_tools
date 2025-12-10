import subprocess
import csv
from io import BytesIO, StringIO
import hashlib
import time

from werkzeug.exceptions import UnsupportedMediaType


def validate_csv(csv_output, video_size):
    """Validate that all rows in the CSV data have exactly three columns,
    correct types, and are within video boundaries."""
    csv_output.seek(0)
    reader = csv.reader(csv_output)

    for row in reader:
        if len(row) != 3:
            # print(f"Invalid row (wrong number of columns): {row}")
            return False

        try:
            offset = int(row[0])
            size = int(row[1])
        except ValueError:
            # print(f"Invalid row (non-integer values): {row}")
            return False

        if len(row[2]) != 1:
            # print(f"Invalid row (non-single character in third column): {row}")
            return False

        if offset < 0 or size <= 0 or offset + size > video_size:
            # print(f"Invalid row (out-of-bounds offset or size): {row}")
            return False

    return True


def sha512hash_video2(video_stream: BytesIO):
    start_time = time.time()
    video_stream.seek(0)

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=pkt_pos,pkt_size,pict_type",
        "-of",
        "csv=p=0",
        "-",
    ]

    try:
        process = subprocess.run(
            command,
            input=video_stream.read(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise UnsupportedMediaType("ffprobe command timed out.")
    except Exception as e:
        raise UnsupportedMediaType(f"An error occurred while running ffprobe: {e}")
    if process.returncode != 0:
        raise UnsupportedMediaType(
            f"ffprobe encountered an error: {process.stderr.decode()}"
        )

    csv_output_data = process.stdout.decode()
    if not csv_output_data.strip():
        raise UnsupportedMediaType("No data returned from ffprobe.")

    video_size = len(video_stream.getvalue())
    csv_output = StringIO(csv_output_data, video_stream.s)
    if not validate_csv(csv_output_data, video_size):
        raise UnsupportedMediaType("CSV data is invalid.")

    csv_output.seek(0)
    reader = csv.reader(csv_output)
    cumulative_hash = hashlib.sha512()
    video_stream.seek(0)

    try:
        for index, row in enumerate(reader):
            if len(row) != 3:
                raise UnsupportedMediaType(f"Skipping malformed row: {row}")
            try:
                offset, size, frame_type = row
                offset = int(offset)
                size = int(size)
            except ValueError:
                raise UnsupportedMediaType(
                    f"Invalid offset or size values in row: {row}"
                )
                continue

            if frame_type == "I":
                try:
                    video_stream.seek(offset)
                    frame_data = video_stream.read(size)
                    cumulative_hash.update(frame_data)
                except (IOError, ValueError) as e:
                    raise UnsupportedMediaType(
                        f"Error processing frame at offset {offset}: {e}"
                    )
    except csv.Error as e:
        raise UnsupportedMediaType(f"Error parsing CSV data: {e}")

    final_hash = cumulative_hash.hexdigest()
    end_time = time.time()
    process_time = end_time - start_time
    return final_hash, process_time


""" 
# Example usage
# Load video data into memory and create a BytesIO object
try:
    with open('input.mp4', 'rb') as file:
        video_stream = io.BytesIO(file.read())
except IOError as e:
    print(f"Error reading video file: {e}")
else:
    # Compute hash
    compute_cumulative_iframe_hash(video_stream)
 """


def sha512hash_video(video_stream: BytesIO):
    start_time = time.time()

    hash_sha512 = hashlib.sha512()
    chunk_size = 8192
    while chunk := video_stream.read(chunk_size):
        hash_sha512.update(chunk)

    # Return the hexadecimal digest of the hash
    hash = hash_sha512.hexdigest()
    end_time = time.time()
    process_time = end_time - start_time
    return hash, process_time
