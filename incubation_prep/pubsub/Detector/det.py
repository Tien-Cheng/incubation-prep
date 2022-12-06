from os import getenv
from typing import Dict, List

import numpy as np
from docarray import Document, DocumentArray
from yolov5 import YOLOv5
from yolov5.models.common import Detections

from component import Component


class YOLODetector(Component):
    """Takes in a frame /image and returns detections"""

    def __init__(
        self,
        weights_or_url: str,
        device: str,
        image_size: int,
        traversal_path: str = "@r",
    ):
        super().__init__()
        self.traversal_path = traversal_path
        self.model = YOLOv5(weights_or_url, device)
        self.image_size = image_size

    def __call__(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        # NOTE: model currently does not support batch inference
        # list only converts first dim to list, not recursively like tolist
        traversed_docs = docs
        frames: List[np.ndarray] = list(traversed_docs.tensors)

        # Either call Triton or run inference locally
        results: Detections = self.model.predict(frames, size=self.image_size)
        for doc, dets in zip(traversed_docs, results.pred):
            # Make every det a match Document
            self.logger.info(dets)
            doc.matches = DocumentArray(
                [
                    Document(
                        tags={
                            "bbox": det[:4].tolist(),  # ltrb format
                            "class_name": int(det[5].item()),
                            "confidence": det[4].item(),
                        }
                    )
                    for det in dets
                    if det.size()[0] != 0
                ]
            )
        return docs


if __name__ == "__main__":
    executor = YOLODetector(
        weights_or_url=getenv("YOLO_WEIGHTS", "yolov5s.pt"),
        device=getenv("YOLO_DEVICE", "0"),
        image_size=int(getenv("IMAGE_SIZE", 640)),
    )
    executor.serve()
