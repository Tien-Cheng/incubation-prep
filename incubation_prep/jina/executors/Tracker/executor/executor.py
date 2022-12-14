import json
from datetime import datetime
from os import getenv
from typing import Dict, List, Optional, Tuple, Union

import torch
import redis
import numpy as np
from confluent_kafka import Producer
from simpletimer import StopwatchKafka

from docarray import Document, DocumentArray
from jina import Executor, requests
from bytetracker import BYTETracker


class ObjectTracker(Executor):
    """"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trackers: Dict[str, BYTETracker] = {}

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
        self.non_triton_timer = StopwatchKafka(
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
    def track(self, docs: DocumentArray, **kwargs):
        # Load tensors if necessary
        blobs = docs.blobs
        if len(docs.find({"uri": {"$exists": True}})) != 0:
            docs.apply(self._load_uri_to_image_tensor)
        elif len(docs.find({"tags__redis": {"$exists": True}})) != 0:
            docs.apply(lambda doc: self._load_image_tensor_from_redis(doc))
        elif blobs is not None:
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
        else:
            self.last_frame[output_stream] = frame_id
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
            all_dets = docs.map(self._get_dets)
            for frame, dets in zip(docs, all_dets):
                if not frame.matches:
                    continue
                output_stream: str = frame.tags["output_stream"]
                if output_stream not in self.trackers:
                    self._create_tracker(output_stream)
                with self.non_triton_timer(
                    metadata={
                        "event": "non_triton_model_processing",
                        "timestamp": datetime.now().isoformat(),
                        "executor": self.executor_name,
                        "executor_id": self.executor_id,
                    }
                ):
                    tracks = self.trackers[output_stream].update(dets, None)
                # Update matches using tracks
                frame.matches = self._update_dets(tracks)
                
            docs.tensors = None
            if blobs is not None:
                docs.blobs = blobs
            return docs

    def _create_tracker(self, name: str):
        tracker = BYTETracker()
        self.trackers[name] = tracker

    @staticmethod
    def _get_dets(frame: Document) -> List[Tuple[List[Union[int, float]], float, str]]:
        det = [
            [
                *det.tags["bbox"],
                det.tags["confidence"],
                det.tags["class_name"]
            ]
            for det in frame.matches
        ]
        return torch.tensor(np.array(det, dtype=np.float32))

    @staticmethod
    def _update_dets(tracks: np.ndarray) -> DocumentArray:
        results = []
        try:
            tracks = list(tracks)
            for track in tracks:
                results.append(
                    Document(
                        tags={
                            "bbox": list(track[:4]),
                            "confidence": track[6],
                            "class_name": track[5],
                            "track_id": track[4],
                        }
                    )
                )
        except Exception as e:
            print(e)
        return DocumentArray(results)

    @staticmethod
    def _load_uri_to_image_tensor(doc: Document) -> Document:
        if doc.uri:
            doc = doc.load_uri_to_image_tensor()
            # Convert channels from NHWC to NCHW
            # doc.tensor = np.transpose(doc.tensor, (2, 1, 0))
        return doc

    def _load_image_tensor_from_redis(self, doc: Document) -> Document:
        image_key = doc.tags["redis"]
        if self.rds.exists(image_key) != 0:
            doc.blob = self.rds.get(image_key)
            # Load bytes
            doc = doc.convert_blob_to_image_tensor()
            doc.pop("blob")
        return doc

