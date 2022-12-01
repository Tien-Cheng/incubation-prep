import cv2
import numpy as np
from time import perf_counter
from jina import Executor, requests
from docarray import DocumentArray, Document
from typing import Optional, List, Tuple, Dict

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
        **kwargs
    ):
        super().__init__(**kwargs)
        self.traversal_path = traversal_path
        self.model = ScaledYOLOV4(
            triton_url=triton_url,
            model_name=model_name,
            model_version=str(model_version),
            max_batch_size=1,
            model_image_size=image_size,
        )
        self.classes = classes

    @requests
    def detect(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        # NOTE: model currently does not support batch inference
        # list only converts first dim to list, not recursively like tolist
        traversed_docs = docs
        frames: List[np.ndarray] = list(traversed_docs.tensors)
        start = perf_counter()
        dets: List[List[Tuple]] = self.model.detect_get_box_in(
            frames, classes=self.classes
        )
        end = perf_counter()
        self.logger.info(f"Time taken to predict frame: {end - start}")
        for doc, det in zip(traversed_docs, dets):
            # Make every det a match Document
            doc.matches = DocumentArray(
                [
                    Document(
                        tags={
                            "bbox": bbox,  # ltrb format
                            "class_name": class_name,
                            "confidence": score,
                        }
                    )
                    for (bbox, score, class_name) in det
                ]
            )
        # Draw detections on Tensor
        if bool(parameters.get("draw_bbox", False)):
            traversed_docs.apply(self._draw_bbox)

    @staticmethod
    def _draw_bbox(frame: Document) -> Document:
        img: np.ndarray = frame.tensor
        dets = frame.matches
        if dets:
            bboxes, scores, classes = dets[
                :, ("tags__bbox", "tags__score", "tags__class_name")
            ]
            for (bbox, score, class_) in zip(bboxes, scores, classes):
                l, t, r, b = bbox
                cv2.rectangle(img, (l, t), (r, b), (255, 255, 0), 1)
                cv2.putText(
                    img,
                    f"{class_} ({score * 100:.2f}%)",
                    (l, t - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                )
            frame.tensor = img
        return frame
