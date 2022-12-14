import json
import socket
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from logging import Logger
from os import getenv
from typing import Dict, Optional
from copy import copy

import redis
from confluent_kafka import Consumer, KafkaError, KafkaException, Producer
from docarray import Document, DocumentArray
from imagezmq import ImageSender
from simpletimer import StopwatchKafka
from zmq_subscriber import VideoStreamSubscriber


class Broker(str, Enum):
    kafka: str = "kafka"
    zmq: str = "zmq"
    none: str = ""


class Component(ABC):

    logger = Logger(__name__)

    conf = {
        "bootstrap.servers": getenv("KAFKA_ADDRESS", "127.0.0.1:9092"),
        "client.id": socket.gethostname(),
        "message.max.bytes": 1000000000,
    }
    metrics_topic = getenv("KAFKA_METRICS_TOPIC", "metrics")
    executor_name = getenv("EXECUTOR_NAME")
    executor_id = executor_name + "-" + datetime.now().isoformat()

    # Set up producer for Kafka metrics
    timer = StopwatchKafka(
        bootstrap_servers=getenv("KAFKA_ADDRESS", "127.0.0.1:9092"),
        kafka_topic=metrics_topic,
        metadata={"executor": executor_name},
        kafka_parition=-1,
    )

    last_frame: Dict[str, str] = {}
    metric_producer = Producer(conf)

    # Redis caching
    try:
        rds = redis.Redis(
            host=getenv("REDIS_HOST", "localhost"),
            port=int(getenv("REDIS_PORT", 6379)),
            db=int(getenv("REDIS_DB", 0)),
        )
    except:
        rds = None

    def __init__(self, msg_broker: Optional[Broker] = None):
        if msg_broker is None:
            msg_broker = getenv("BROKER", Broker.kafka)
        self.broker = msg_broker

        if msg_broker == Broker.kafka:
            producer_conf = {**self.conf}
            consumer_conf = {
                **self.conf,
                "group.id": getenv("KAFKA_CONSUMER_GROUP", "foo"),
                "auto.offset.reset": "smallest",
                "fetch.max.bytes": 1000000000,
                "max.partition.fetch.bytes": 1000000000,
            }
            self.produce_topic = getenv("KAFKA_PRODUCE_TOPIC", None)
            self.consume_topic = getenv("KAFKA_CONSUME_TOPIC", None)
            if self.produce_topic:
                self.producer = Producer(producer_conf)
            if self.consume_topic:
                self.consumer = Consumer(consumer_conf)
        elif msg_broker == Broker.zmq:
            self.consumer = VideoStreamSubscriber(
                hostname=getenv("ZMQ_HOSTNAME", "*"), port=getenv("ZMQ_PORT_IN", "5555")
            )
            self.producer = ImageSender(
                f"tcp://{getenv('ZMQ_HOSTNAME', '*')}:{getenv('ZMQ_PORT_OUT', '5556')}",
                REQ_REP=False,
            )
        else:
            self.producer = None
            self.consumer = None

    @abstractmethod
    def __call__(
        self, data: DocumentArray, parameters: Optional[dict] = {}, **kwargs
    ) -> DocumentArray:
        return data

    def serve(self, filter_stream: Optional[str] = None):
        """
        This is a wrapper around __call__ that will do the following:

        1. Consume using either Kafka or ZMQ
        2. Process consumed data into DocArray
        3. Pass DocArray to __call__
        4. Process output
        5. Produce

        The choice of processor will depend on an environment variable
        """
        if self.broker == Broker.zmq:
            self._process_zmq(filter_stream)
        elif self.broker == Broker.kafka:
            self._process_kafka(filter_stream)

    def _process_zmq(self, filter_stream: Optional[str] = None):
        try:
            while True:
                # Get frames
                data = self.consumer.receive(timeout=1)
                if data is None:
                    continue
                # Convert metadata to docarray
                assert isinstance(data, bytes), "Is byte"
                frame_docs = DocumentArray.from_bytes(data)
                if filter_stream is not None:
                    frame_docs = frame_docs.find(
                        {"tags__output_stream": {"$eq": filter_stream}}, limit=None
                    )
                if len(frame_docs) == 0:
                    continue  # skip frames if pod not meant to receive them
                result = self._call_main(frame_docs, frame_docs.blobs is not None)
                # Process Results
                if self.producer:
                    self.producer.zmq_socket.send(result.to_bytes())
        except (SystemExit, KeyboardInterrupt):
            print("Exit due to keyboard interrupt")
        except Exception as ex:
            print("Python error with no Exception handler:")
            print("Traceback error:", ex)
            traceback.print_exc()
        finally:
            self.consumer.close()

    def _process_kafka(self, filter_stream: Optional[str] = None):
        try:
            if not self.consume_topic:
                raise ValueError("No consumer topic set!")
            self.consumer.subscribe([self.consume_topic])
            while True:
                # Get frames
                data = self.consumer.poll(timeout=1)
                if data is None:
                    continue
                if data.error():
                    if data.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        raise KafkaException(data.error())
                # Convert metadata to docarray
                frame_docs = DocumentArray.from_bytes(data.value())
                if filter_stream is not None:
                    frame_docs = frame_docs.find(
                        {"tags__output_stream": {"$eq": filter_stream}}, limit=None
                    )
                if len(frame_docs) == 0:
                    continue  # skip frames if pod not meant to receive them
                # Check that frame contains tensors
                result = self._call_main(frame_docs, frame_docs.blobs is not None)
                # Process Results
                if self.produce_topic:
                    self.producer.produce(self.produce_topic, value=result.to_bytes())
                    self.producer.poll(0)
        except (SystemExit, KeyboardInterrupt):
            print("Exit due to keyboard interrupt")
        except Exception as ex:
            print("Python error with no Exception handler:")
            print("Traceback error:", ex)
            traceback.print_exc()
        finally:
            self.consumer.close()

    def _call_main(
        self, docs: DocumentArray, send_tensors: bool = False # TODO: Remove from client side to avoid error
    ) -> DocumentArray:
        blobs = docs.blobs
        if not bool(blobs[0]):
            blobs = None # For some reason if blobs is none, will be [b""]
        # assert blobs is None, f"TEST REDIS WORKS: {blobs}"
        if blobs is None:
            # tmp1 = False
            if len(docs[...].find({"uri": {"$exists": True}})) != 0:
                docs.apply(self._load_uri_to_image_tensor)
            elif len(docs[...].find({"tags__redis": {"$exists": True}})) != 0:
                # tmp1 = True
                docs.apply(lambda doc: self._load_image_tensor_from_redis(doc))
            # assert tmp1, "TEST APPLY"
        else:
            docs.apply(self._load_image_tensor_from_blob)

        # NOTE: Assume only 1 doc inside docarray
        # Check for dropped frames
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
                "event": "overall",
                "frame_id": frame_id,
                "video_path": video_source,
                "output_stream": output_stream,
                "timestamp": datetime.now().isoformat(),
                "executor": self.executor_name,
                "executor_id": self.executor_id,
            }
        ):
            docs = self.__call__(docs)
        # No matter what, remove the tensors
        # even if sending frame over kafka, I want
        # to store it in blob as jpeg (for compression)
        # so I don't need the tensor
        docs.tensors = None
        docs.blobs = blobs
        return docs

    @staticmethod
    def _load_uri_to_image_tensor(doc: Document) -> Document:
        if doc.uri:
            doc = doc.load_uri_to_image_tensor()
            # NOTE: Testing shows not necessary and actually breaks stuff
            # Convert channels from NHWC to NCHW
            # doc.tensor = np.transpose(doc.tensor, (2, 1, 0))
        return doc

    @staticmethod
    def _load_image_tensor_from_blob(doc: Document) -> Document:
        if doc.blob:
            doc = doc.convert_blob_to_image_tensor()
        return doc

    def _load_image_tensor_from_redis(self, doc: Document) -> Document:
        if "redis" in doc.tags:
            image_key = doc.tags["redis"]
            if self.rds.exists(image_key) != 0:
                doc.blob = self.rds.get(image_key)
                # Load bytes
                doc = doc.convert_blob_to_image_tensor()
                assert doc.tensor is not None, "Redis converts successfully"

        else:
            self.logger.error("Failed to load from redis")
        return doc
