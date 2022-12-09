import json
from datetime import datetime
from os import getenv
from typing import Dict, List

import numpy as np
from confluent_kafka import Producer
from docarray import Document, DocumentArray
from simpletimer import StopwatchKafka
from yolov5 import YOLOv5
from yolov5.models.common import Detections

from jina import Executor, requests


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
        self.is_triton = weights_or_url.startswith("grpc")

        # Common Logic
        self.metrics_topic = getenv("KAFKA_METRICS_TOPIC", "metrics")
        self.executor_name = self.__class__.__name__
        self.executor_id = self.executor_name + "-" + datetime.now().isoformat()
        conf = {
            "bootstrap.servers": getenv("KAFKA_ADDRESS", "127.0.0.1:9092"),
            "client.id": self.executor_id,
            "message.max.bytes": 1000000000,
        }
        self.timer = StopwatchKafka(
            bootstrap_servers=getenv("KAFKA_ADDRESS", "127.0.0.1:9092"),
            kafka_topic=self.metrics_topic,
            metadata={"executor": self.executor_name},
            kafka_parition=-1,
        )

        self.last_frame: Dict[str, str] = {}
        self.metric_producer = Producer(conf)

    @requests
    def detect(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        # Load tensors if necessary
        send_tensors = True
        if (
            docs[...].tensors is None
            and len(docs[...].find({"uri": {"$exists": True}})) != 0
        ):
            send_tensors = False
            docs[...].apply(self._load_uri_to_image_tensor)

        # Check for dropped frames ( assume only 1 doc )
        frame_id = docs[0].tags["frame_id"]
        output_stream = docs[0].tags["output_stream"]
        video_source = docs[0].tags["video_path"]
        if output_stream not in self.last_frame:
            self.last_frame[output_stream] = frame_id
        if frame_id < self.last_frame[output_stream]:
            self.metric_producer.produce(
                self.metrics_topic,
                value=json.dumps(
                    {
                        "type": "dropped_frame",
                        "timestamp": datetime.now().isoformat(),
                        "executor": self.executor_name,
                        "executor_id": self.executor_id,  # contain time the exec was initialized
                        "frame_id": frame_id,
                        "output_stream": output_stream,
                        "video_source": video_source,
                    }
                ).encode("utf-8"),
            )
            self.metric_producer.poll(0)
            self.logger.warn("Dropped frame")
        # NOTE: model currently does not support batch inference
        # list only converts first dim to list, not recursively like tolist
        with self.timer(
            metadata={
                "frame_id": frame_id,
                "video_path": video_source,
                "output_stream": output_stream,
                "timestamp": datetime.now().isoformat(),
                "executor": self.executor_name,
                "executor_id": self.executor_id,
            }
        ):
            traversed_docs = docs
            frames: List[np.ndarray] = list(traversed_docs.tensors)
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
                    results: Detections = self.model.predict(
                        frames, size=self.image_size
                    )
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
            if not send_tensors:
                docs[...].tensors = None
            return docs

    @staticmethod
    def _load_uri_to_image_tensor(doc: Document) -> Document:
        if doc.uri:
            doc = doc.load_uri_to_image_tensor()
            # Convert channels from NHWC to NCHW
            # doc.tensor = np.transpose(doc.tensor, (2, 1, 0))
        return doc
