from typing import List, Optional, Tuple

import numpy as np
import pkg_resources
from scaledyolov4.scaled_yolov4 import ScaledYOLOV4


class ObjectDetector:
    def __init__(self, image_size: int = 896, classes: Optional[List[str]] = None):
        self.model = ScaledYOLOV4(
            bgr=True,
            model_image_size=image_size,
            max_batch_size=1,
            half=True,
            same_size=True,
            weights=pkg_resources.resource_filename(
                "scaledyolov4", "weights/yolov4-p6_-state.pt"
            ),
            cfg=pkg_resources.resource_filename(
                "scaledyolov4", "weights/yolov4-p6.yaml"
            ),
        )
        self.classes = None

    def detect(self, frames: List[np.ndarray]) -> List[List[Tuple]]:
        dets = self.model.detect_get_box_in(
            frames, box_format="ltwh", classes=self.classes
        )
        return dets
