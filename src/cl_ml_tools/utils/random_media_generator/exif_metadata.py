import subprocess
from datetime import datetime

from pydantic import BaseModel, Field


class ExifMetadata(BaseModel):
    MIMEType: str
    CreateDate: datetime | None = None
    UserComments: list[str] = Field(default_factory=list)

    # Internal / runtime-only fields
    cmd: list[str] = Field(
        default_factory=lambda: ["exiftool", "-q", "-m"],
        exclude=True,
    )
    has_metadata: bool = Field(default=False, exclude=True)

    def updateCreateDate(self) -> None:
        if self.CreateDate is None:
            return

        self.has_metadata = True
        date_str_exif = self.CreateDate.strftime("%Y:%m:%d %H:%M:%S")

        if self.MIMEType.startswith("image/"):
            self.cmd.extend(
                [
                    f"-DateTimeOriginal={date_str_exif}",
                    f"-CreateDate={date_str_exif}",
                    f"-ModifyDate={date_str_exif}",
                    f"-FileCreateDate={date_str_exif}",
                    f"-FileModifyDate={date_str_exif}",
                ]
            )

        elif self.MIMEType.startswith("video/"):
            self.cmd.extend(
                [
                    f"-QuickTime:CreateDate={date_str_exif}",
                    f"-QuickTime:ModifyDate={date_str_exif}",
                    f"-QuickTime:TrackCreateDate={date_str_exif}",
                    f"-QuickTime:TrackModifyDate={date_str_exif}",
                    f"-QuickTime:MediaCreateDate={date_str_exif}",
                    f"-QuickTime:MediaModifyDate={date_str_exif}",
                    f"-Keys:CreationDate={date_str_exif}",
                    f"-FileCreateDate={date_str_exif}",
                    f"-FileModifyDate={date_str_exif}",
                ]
            )

    def updateUserComments(self) -> None:
        if not self.UserComments:
            return

        self.has_metadata = True

        for comment in self.UserComments:
            self.cmd.extend(
                [
                    f"-UserComment={comment}",
                    f"-Comment={comment}",
                    f"-XMP-dc:Description={comment}",
                    f"-EXIF:ImageDescription={comment}",
                ]
            )

            if self.MIMEType.startswith("video/"):
                self.cmd.append(f"-QuickTime:Comment={comment}")

    def write(self, filepath: str) -> None:
        self.updateCreateDate()
        self.updateUserComments()

        if not self.has_metadata:
            print("Metadata is empty")
            return

        self.cmd.extend(["-overwrite_original", filepath])

        try:
            _ = subprocess.run(
                self.cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            print("Metadata written successfully.")
        except subprocess.CalledProcessError as e:
            stderr_obj: object = e.stderr  # pyright: ignore[reportAny]
            stderr_str: str

            if isinstance(stderr_obj, (bytes, bytearray)):
                stderr_str = stderr_obj.decode(errors="replace")
            else:
                stderr_str = str(stderr_obj)

            raise Exception(f"Error calling ExifTool:\n{stderr_str}") from e

        except FileNotFoundError as e:
            raise Exception("ExifTool not found") from e

        except Exception:
            print(f"Warning: Failed to write metadata for {self.MIMEType}")
