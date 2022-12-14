import json
from datetime import datetime
from os import getenv
from pathlib import Path
from typing import Dict

import cv2
import redis
from confluent_kafka import Producer
from simpletimer import StopwatchKafka

from jina import Document, DocumentArray, Executor, requests


class SaveStream(Executor):
    """"""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        path: str = "./final",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.path = path

        Path(path).mkdir(exist_ok=True, parents=True)

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

        # Redis caching
        try:
            self.rds = redis.Redis(
                host=getenv("REDIS_HOST", "localhost"),
                port=int(getenv("REDIS_PORT", 6379)),
                db=int(getenv("REDIS_DB", 0)),
            )
        except:
            self.rds = None

    @requests
    def save(self, docs: DocumentArray, **kwargs):
        """Read frames and save them in NFS or Redis

        :param docs: _description_
        :type docs: DocumentArray
        """
        blobs = docs.blobs
        if len(docs.find({"uri": {"$exists": True}})) != 0:
            docs.apply(self._load_uri_to_image_tensor)
        elif len(docs.find({"tags__redis": {"$exists": True}})) != 0:
            docs.apply(lambda doc: self._load_image_tensor_from_redis(doc))
        else:
            docs.apply(lambda doc : doc.convert_blob_to_image_tensor())
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

            for frame in docs:
                # Get stream name
                if frame.matches:
                    bboxes, scores, classes, track_ids = frame.matches[
                        :,
                        (
                            "tags__bbox",
                            "tags__confidence",
                            "tags__class_name",
                            "tags__track_id",
                        ),
                    ]
                    for (bbox, score, class_, id_) in zip(
                        bboxes, scores, classes, track_ids
                    ):
                        l, t, r, b = tuple(map(int, bbox))
                        cv2.rectangle(frame.tensor, (l, t), (r, b), (0, 0, 255), 2)
                        cv2.putText(
                            frame.tensor,
                            f"[ID: {id_}] {class_} ({score * 100:.2f}%)",
                            (l, t - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),
                        )
                frame.tensor = cv2.resize(frame.tensor, (self.width, self.height))
                # Save
                filename = f"video-{frame.tags['output_stream']}-frame-{int(frame.tags['frame_id'])}-{datetime.now().isoformat()}.jpg"
                path = f"{self.path}/{filename}"
                self.logger.info(f"Saving to {path}")
                # We assume input is RGB
                frame.tensor = cv2.cvtColor(frame.tensor, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(str(path), frame.tensor)
                if not success:
                    self.logger.error("Failed to save file")
            docs.tensors = None
            if blobs is not None:
                docs.blobs = blobs
            return docs
    @staticmethod
    def _load_uri_to_image_tensor(doc: Document) -> Document:
        if doc.uri:
            doc = doc.load_uri_to_image_tensor()
            # Convert channels from NHWC to NCHW
            # doc.tensor = np.transpose(doc.tensor, (2, 1, 0))
        return doc

    def _load_image_tensor_from_redis(self, doc: Document) -> Document:
        if "redis" in doc.tags:
            image_key = doc.tags["redis"]
            if self.rds.exists(image_key) != 0:
                doc.blob = self.rds.get(image_key)
                # Load bytes
                doc = doc.convert_blob_to_image_tensor()
                doc.pop("blob")
        return doc

