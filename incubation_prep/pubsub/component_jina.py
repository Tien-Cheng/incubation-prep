import json
import socket
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from logging import Logger
from os import getenv
from typing import Dict, Optional

import numpy as np
from confluent_kafka import Consumer, KafkaError, KafkaException, Producer
from docarray import Document, DocumentArray
from imagezmq import ImageSender
from simpletimer import StopwatchKafka
from zmq_subscriber import VideoStreamSubscriber

from .component import Component


class Broker(str, Enum):
    kafka: str = "kafka"
    zmq: str = "zmq"
    none: str = ""


class JinaComponent(Component):

    logger = Logger(__name__)

    conf = {
        "bootstrap.servers": getenv("KAFKA_ADDRESS", "127.0.0.1:9092"),
        "client.id": socket.gethostname(),
        "message.max.bytes": 1000000000,
    }
    metrics_topic = getenv("KAFKA_METRICS_TOPIC", "metrics")
    executor_name = getenv("EXECUTOR_NAME")

    # Set up producer for Kafka metrics
    timer = StopwatchKafka(
        bootstrap_servers=getenv("KAFKA_ADDRESS", "127.0.0.1:9092"),
        kafka_topic=metrics_topic,
        metadata={"type": "processing_time", "executor": executor_name},
        kafka_parition=-1,
    )

    last_frame: Dict[str, str] = {}
    metric_producer = Producer(conf)

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

    def serve(self, send_tensors: bool = True):
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
            self._process_zmq(send_tensors)
        elif self.broker == Broker.kafka:
            self._process_kafka(send_tensors)

    def _process_zmq(self, send_tensors):
        try:
            while True:
                # Get frames
                data = self.consumer.receive(timeout=1)
                if data is None:
                    continue
                # Convert metadata to docarray
                assert isinstance(data, bytes), "Is byte"
                frame_docs = DocumentArray.from_bytes(data)
                result = self._call_main(frame_docs, send_tensors)
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

    def _process_kafka(self, send_tensors):
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
                result = self._call_main(frame_docs, send_tensors)
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
        self, docs: DocumentArray, send_tensors: bool = True
    ) -> DocumentArray:
        if not send_tensors:
            if docs[..., "uri"] is not None:
                docs[...].apply(self._load_uri_to_image_tensor)

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
                        "frame_id": frame_id,
                        "output_stream": output_stream,
                        "video_source": video_source,
                    }
                ).encode("utf-8"),
            )
            self.metric_producer.poll(0)

        # If dropped, go call metric producer
        with self.timer(
            metadata={
                "frame_id": frame_id,
                "video_path": video_source,
                "output_stream": output_stream,
            }
        ):
            # TODO: Measure time and dropped frames here
            docs = self.__call__(docs)
        if not send_tensors:
            docs[...].tensors = None
        return docs

    @staticmethod
    def _load_uri_to_image_tensor(doc: Document) -> Document:
        if doc.uri:
            doc = doc.load_uri_to_image_tensor()
            # Convert channels from NHWC to NCHW
            doc.tensor = np.transpose(doc.tensor, (2, 1, 0))
        return doc