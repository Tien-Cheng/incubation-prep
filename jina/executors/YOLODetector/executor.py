import numpy as np
from jina import Executor, requests
from docarray import DocumentArray, Document
from typing import Optional, List, Tuple

from scaledyolov4.scaled_yolov4 import ScaledYOLOV4


class YOLODetector(Executor):
    """Takes in a frame /image and returns detections"""

    def __init__(
        self,
        triton_url: str,
        model_name: str,
        model_version: str,
        image_size: int,
        traversal_path: str = "@r",
        classes: Optional[List[str]] = None,
    ):
        self.traversal_path = traversal_path
        self.model = ScaledYOLOV4(
            triton_url=triton_url,
            model_name=model_name,
            model_version=model_version,
            max_batch_size=1,
            model_image_size=image_size,
        )
        self.classes = classes

    @requests
    def detect(self, docs: DocumentArray, **kwargs):
        # NOTE: model currently does not support batch inference
        # list only converts first dim to list, not recursively like tolist
        traversed_docs = docs[self.traversal_path]
        frames: List[np.ndarray] = list(traversed_docs.tensors)
        dets: List[List[Tuple]] = self.model.detect_get_box_in(
            frames, classes=self.classes
        )
        for doc, det in zip(traversed_docs, dets):
            # Make every det a match Document
            doc.matches = DocumentArray(
                [
                    Document(
                        tags={
                            "bbox": bbox,  # ltrb format
                            "class_name": class_name,
                            "score": score,
                        }
                    )
                    for (bbox, score, class_name) in det
                ]
            )
