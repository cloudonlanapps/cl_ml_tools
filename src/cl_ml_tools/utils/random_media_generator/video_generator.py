from dataclasses import dataclass, field
import os
from typing import List, Optional

import cv2

from .base_media import BaseMedia
from .scene_generator import SceneGenerator
from .errors import JSONValidationError


@dataclass
class VideoGenerator(BaseMedia):
    scenes: List[SceneGenerator] = field(default_factory=list)
    fps: Optional[int] = 30

    @classmethod
    def from_dict(cls,out_dir:str,  data: dict):
        
        processedData = data.copy()
        processedData["out_dir"] = out_dir
        # Process base fields first, get them as a dictionary
        base_fields_dict = BaseMedia.from_dict(processedData).__dict__
        scenes_list = []
        if "scenes" in data and isinstance(data["scenes"], list):
            scenes_list = [
                SceneGenerator.from_dict(scene_data) for scene_data in data["scenes"]
            ]
        elif "scenes" not in data:
            raise JSONValidationError("'scenes' not found for VideoDescription.")
        else:
            raise JSONValidationError("Invalid 'scenes' data for VideoDescription.")

        init_args = {
            **base_fields_dict, 
            "scenes": scenes_list,  
            "fps": data.get("fps")
        }
        return cls( **init_args)

    def to_dict(self) -> dict:
        """Converts the VideoDescription instance to a dictionary for JSON serialization."""
        data = super().to_dict()
        data["scenes"] = [scene.to_dict() for scene in self.scenes]
        data["fps"] = self.fps
        return data

    
    

    def generate(self):
        out = cv2.VideoWriter(
            self.temp_filepath, self.fourcc_code, self.fps, (self.width, self.height)
        )
        if not out.isOpened():
            raise Exception(f"Error: Could not open video writer for {self.filepath}")

        for scene_idx, scene in enumerate(self.scenes):
            #print(f"  Rendering Scene {scene_idx + 1} (Duration: {scene.duration}s)")
            scene.generate(out=out, fps=self.fps, width=self.width, height=self.height)

        out.release()
        print(f"Video '{self.fileName}' created by OpenCV.")
        self.update_metadata()
        os.rename(self.temp_filepath, self.filepath)
       
