from io import BytesIO
from PIL import Image, ImageFile
import hashlib
from pillow_heif import register_heif_opener
import time

# TODO: Do we need to record if the image is truncated?
ImageFile.LOAD_TRUNCATED_IMAGES = True
register_heif_opener()


def sha512hash_image(image_stream: BytesIO):
    start_time = time.time()
    with Image.open(image_stream) as im:
        hash = hashlib.sha512(im.tobytes()).hexdigest()
    end_time = time.time()
    process_time = end_time - start_time
    return hash, process_time
