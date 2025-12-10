import subprocess
from dataclasses import field
from datetime import datetime
from typing import List, Optional


class ExifMetadata:
    def __init__(
        self,
        MIMEType: str,
        CreateDate: Optional[datetime] = None,
        UserComments: List[str] = field(default_factory=list),
    ):
        self.MIMEType = MIMEType
        self.CreateDate = CreateDate
        self.UserComments = UserComments
        self.cmd = ["exiftool", "-q", "-m"]
        self.has_metadata = False
        pass

    def updateCreateDate(self):
        if self.CreateDate is None:
            return
        self.has_metadata = True
        date_str_exif = self.CreateDate.strftime("%Y:%m:%d %H:%M:%S")
        if self.MIMEType.startswith("image/"):
            self.cmd.extend(
                [
                    f"-DateTimeOriginal={date_str_exif}",  # For photos
                    f"-CreateDate={date_str_exif}",  # General image creation date
                    f"-ModifyDate={date_str_exif}",  # Modification date
                    f"-FileCreateDate={date_str_exif}",  # File system creation date
                    f"-FileModifyDate={date_str_exif}",  # File system modification date
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
                    f"-Keys:CreationDate={date_str_exif}",  # Newer MP4 (ISO BMFF) tag
                    f"-FileCreateDate={date_str_exif}",
                    f"-FileModifyDate={date_str_exif}",
                ]
            )

    def updateUserComments(self):
        if self.UserComments:
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

    def write(self, filepath: str):
        self.updateCreateDate()
        self.updateUserComments()
        if not self.has_metadata:
            print("Matadata is Empty")
            return
        self.cmd.extend(["-overwrite_original", filepath])
        try:
            try:
                result = subprocess.run(
                    self.cmd, capture_output=True, text=True, check=True
                )
                if result.returncode != 0:
                    raise Exception(f"ExifTool failed. \nstderr:\n{result.stderr}")
                
                print("Matadata is writen successfully. ")

            except subprocess.CalledProcessError as e:
                raise Exception(f"Error calling ExifTool: {e}")
            except FileNotFoundError:
                raise Exception("ExifTool not found")
        except Exception:
            print(f"Warning: Failed to write Metadata for {self.MIMEType}")



