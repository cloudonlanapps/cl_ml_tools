import hashlib
from io import BytesIO

from marshmallow import ValidationError


def get_md5_hexdigest(bytes_io: BytesIO):
    hash_md5 = hashlib.md5()
    bytes_io.seek(0)

    for chunk in iter(lambda: bytes_io.read(4096), b""):
        hash_md5.update(chunk)

    return hash_md5.hexdigest()


def validate_md5String(bytes_io: BytesIO, md5String: str):
    hash_md5 = hashlib.md5()

    bytes_io.seek(0)

    for chunk in iter(lambda: bytes_io.read(4096), b""):
        hash_md5.update(chunk)

    if hash_md5.hexdigest() == md5String:
        return

    raise ValidationError(
        {
            "md5String": ["md5String provided is not matching with media!"],
        }
    )
