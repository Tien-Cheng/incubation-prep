import numpy as np
from jina import Executor, requests
from docarray import Document, DocumentArray

import cv2

class FrameDrawer(Executor):
    """"""
    @requests
    def render(self, docs: DocumentArray, **kwargs):
        docs.apply(self._draw_bbox)

    @staticmethod
    def _draw_bbox(frame: Document) -> Document:
        img: np.ndarray = frame.tensor
        dets = frame.matches
        if dets:
            bboxes, scores, classes = dets[
                :, ("tags__bbox", "tags__score", "tags__class_name")
            ]
            for (bbox, score, class_) in zip(bboxes, scores, classes):
                l, t, r, b = tuple(map(int,bbox))
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