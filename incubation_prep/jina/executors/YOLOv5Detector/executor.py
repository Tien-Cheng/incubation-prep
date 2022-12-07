from os import getenv
from time import perf_counter
from typing import Dict, List

import numpy as np
from docarray import Document, DocumentArray
from yolov5 import YOLOv5
from yolov5.models.common import Detections

from jina import Executor, requests
from simpletimer import StopwatchKafka


class YOLODetector(Executor):
    """Takes in a frame /image and returns detections"""

    def __init__(
        self,
        weights_or_url: str,
        device: str,
        image_size: int,
        traversal_path: str = "@r",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.traversal_path = traversal_path
        self.model = YOLOv5(weights_or_url, device)
        self.image_size = image_size
        metrics_topic = getenv("KAFKA_METRICS_TOPIC", "metrics")
        executor_name = getenv("EXECUTOR_NAME")
        self.timer = StopwatchKafka(
            bootstrap_servers=getenv("KAFKA_ADDRESS", "127.0.0.1:9092"),
            kafka_topic=metrics_topic,
            metadata={"type": "processing_time", "executor": executor_name},
            kafka_parition=-1,
        )

    @requests
    def detect(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        # NOTE: model currently does not support batch inference
        # list only converts first dim to list, not recursively like tolist
        with self.timer:
            traversed_docs = docs
            frames: List[np.ndarray] = list(traversed_docs.tensors)

            start = perf_counter()
            # Either call Triton or run inference locally
            # Assume RGB image
            results: Detections = self.model.predict(frames, size=self.image_size)
            end = perf_counter()
            self.logger.info(f"Time taken to predict frame: {end - start}")
            for doc, dets in zip(traversed_docs, results.pred):
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
