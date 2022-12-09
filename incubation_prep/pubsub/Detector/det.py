import json
from datetime import datetime
from os import getenv
from typing import Dict, List

import numpy as np
from component import Component
from docarray import Document, DocumentArray
from yolov5 import YOLOv5
from yolov5.models.common import Detections


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
        self.model.model.conf = 0.6
        self.model.model.max_det = 500
        self.image_size = image_size
        self.is_triton = weights_or_url.startswith("grpc")

    def __call__(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        # NOTE: model currently does not support batch inference
        # list only converts first dim to list, not recursively like tolist
        frames: List[np.ndarray] = list(docs.tensors)
        # Either call Triton or run inference locally
        # assumption: image sent is RGB
        if self.is_triton:
            results: Detections = self.model.predict(frames, size=self.image_size)
        else:
            with self.timer(
                metadata={
                    "event": "non_triton_model_processing",
                    "timestamp": datetime.now().isoformat(),
                    "executor": self.executor_name,
                    "executor_id": self.executor_id,
                }
            ):
                results: Detections = self.model.predict(frames, size=self.image_size)
            # Track detections by model
            base_metric = {
                "type": "non_triton_inference",
                "timestamp": datetime.now().isoformat(),
                "executor": self.executor_name,
                "executor_id": self.executor_id,
                "value": 1,
            }
            for frame in docs:
                metric = {
                    **base_metric,
                    "output_stream": frame.tags["output_stream"],
                    "video_source": frame.tags["video_path"],
                    "frame_id": frame.tags["frame_id"],
                }
                # Produce metric
                self.metric_producer.produce(
                    self.metrics_topic, value=json.dumps(metric).encode("utf-8")
                )
                self.metric_producer.poll(0)
        for doc, dets in zip(docs, results.pred):
            # Make every det a match Document
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
