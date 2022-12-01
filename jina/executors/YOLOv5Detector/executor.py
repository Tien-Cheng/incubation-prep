import numpy as np
from time import perf_counter
from jina import Executor, requests
from docarray import DocumentArray, Document
from typing import List, Dict

from yolov5 import YOLOv5
from yolov5.models.common import Detections

class YOLODetector(Executor):
    """Takes in a frame /image and returns detections"""

    def __init__(
        self,
        weights_or_url: str,
        device: str,
        image_size: int,
        traversal_path: str = "@r",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.traversal_path = traversal_path
        self.model = YOLOv5(weights_or_url, device)
        self.image_size = image_size

    @requests
    def detect(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        # NOTE: model currently does not support batch inference
        # list only converts first dim to list, not recursively like tolist
        traversed_docs = docs
        frames: List[np.ndarray] = list(traversed_docs.tensors)
        
        start = perf_counter()
        # Either call Triton or run inference locally
        results: Detections = self.model.predict(frames, size=self.image_size)
        end = perf_counter()
        self.logger.info(f"Time taken to predict frame: {end - start}")
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
                    for det in dets if det.size()[0] != 0
                ]
            )