import json
from datetime import datetime
from os import getenv
from typing import Dict, Optional, Union
from io import BytesIO

import redis
import cv2
import numpy as np

from confluent_kafka import Producer
from simpletimer import StopwatchKafka
from vidgear.gears import NetGear, WriteGear

from jina import Executor, requests
from docarray import DocumentArray, Document


class StreamOutput(Executor):
    """"""

    def __init__(
        self,
        address: str = "rtsp://localhost",
        port: str = "8554",
        fps: int = 30,
        width: int = 1280,
        height: int = 720,
        zmq: bool = False,
        ffmpeg_config: Optional[Dict] = None,
        zmq_config: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.address = address
        self.port = port
        self.fps = fps
        self.width = width
        self.height = height
        self.zmq = zmq
        if ffmpeg_config is None:
            ffmpeg_config = {
                "-preset:v": "ultrafast",
                "-tune": "zerolatency",
                "-f": "rtsp",
                "-rtsp_transport": "tcp",
            }
        self.ffmpeg_config = ffmpeg_config
        if zmq_config is None:
            zmq_config = {
                "jpeg_compression": True,
                "jpeg_compression_quality": 90,
                "jpeg_compression_fastdct": True,
                "jpeg_compression_fastupsample": True,
            }
        self.zmq_config = zmq_config
        self.streams: Dict[str, Union[NetGear, WriteGear]] = {}

        if zmq:
            self.create_stream = self.create_stream_zmq

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

    def create_stream(self, name: str):
        self.streams[name] = WriteGear(
            f"{self.address}:{self.port}/{name}",
            logging=True,
            compression_mode=True,
            **self.ffmpeg_config,
        )

    def create_stream_zmq(self, name):
        self.streams[name] = NetGear(
            address=self.address,
            port=self.port,
            logging=True,
            receive_mode=False,
            **self.zmq_config,
        )

    @requests
    def produce(self, docs: DocumentArray, **kwargs):
        """Read frames and send them to their respective output streams

        :param docs: _description_
        :type docs: DocumentArray
        """
        blobs = docs.blobs
        if len(docs.find({"uri": {"$exists": True}})) != 0:
            docs.apply(self._load_uri_to_image_tensor)
        elif len(docs.find({"tags__redis": {"$exists": True}})) != 0:
            docs.apply(lambda doc: self._load_image_tensor_from_redis(doc))
        elif blobs is not None and docs.tensors is None:
            docs.apply(lambda doc : doc.convert_blob_to_image_tensor() if doc.tensor is None else doc)
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
            self.logger.warning("Dropped frame")
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
            for frame in docs:
                # Get stream name
                output_stream: str = frame.tags["output_stream"]
                # Get frame ID
                frame_id = frame.tags["frame_id"]
                if output_stream not in self.streams:
                    self.create_stream(output_stream)
                # VidGear will handle threading for us
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
                        cv2.rectangle(frame.tensor, (l, t), (r, b), (255, 0, 0), 2)
                        cv2.putText(
                            frame.tensor,
                            f"[ID: {id_}] {class_} ({score * 100:.2f}%)",
                            (l, t - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),  # BGR
                        )
                # We assume input is RGB
                frame.tensor = cv2.resize(frame.tensor, (self.width, self.height))
                frame.tensor = cv2.cvtColor(frame.tensor, cv2.COLOR_RGB2BGR)
                try:
                    if self.zmq:
                        self.streams[output_stream].send(frame.tensor)
                    else:
                        self.streams[output_stream].write(frame.tensor)
                except ValueError:
                    if self.zmq:
                        self.create_stream_zmq(output_stream)
                    else:
                        self.create_stream(output_stream)
                    pass
        
            # Don't clear tensor if encoding using numpy
            # and not sending tensor via nfs or redis
            if not docs[0].tags["numpy"]:
                if not (not docs[0].uri and "redis" not in docs[0].tags):
                    docs.tensors = None
                if blobs is not None:
                    docs.blobs = blobs
        return docs

    @staticmethod
    def _load_uri_to_image_tensor(doc: Document) -> Document:
        if doc.uri:
            if doc.tags["numpy"]:
                doc.tensor = np.load(doc.uri)
            else:
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
                if doc.tags["numpy"]:
                    doc.tensor = np.load(BytesIO(doc.blob), allow_pickle=True)
                else:
                    doc = doc.convert_blob_to_image_tensor()
                doc.pop("blob")
        return doc

