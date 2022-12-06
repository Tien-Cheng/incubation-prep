import traceback
import socket

from abc import ABC, abstractmethod
from enum import Enum
from logging import Logger
from typing import Optional
from os import getenv

from docarray import DocumentArray

from imagezmq import ImageSender
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException

from zmq_subscriber import VideoStreamSubscriber


class Broker(str, Enum):
    kafka: str = "kafka"
    zmq: str = "zmq"
    none: str = ""


class Component(ABC):

    logger = Logger(__name__)

    conf = {
        "bootstrap.servers": getenv("KAFKA_ADDRESS", "localhost:9092"),
        "client.id": socket.gethostname(),
        "message.max.bytes": 1000000000,
    }

    # Set up producer for Kafka metrics
    metric_producer = Producer(conf)

    def __init__(self, msg_broker: Optional[Broker] = None):
        if msg_broker is None:
            msg_broker = getenv("BROKER", Broker.kafka)
        self.broker = msg_broker

        if msg_broker == Broker.kafka:
            producer_conf = {**self.conf}
            consumer_conf = {
                **self.conf,
                # "group.id": getenv("KAFKA_CONSUMER_GROUP", "foo"),
                "auto.offset.reset": "smallest",
                "fetch.max.bytes": 1000000000,
                "max.partition.fetch.bytes": 1000000000,
            }
            self.producer = Producer(producer_conf)
            self.consumer = Consumer(consumer_conf)
            self.produce_topic = getenv("KAFKA_PRODUCE_TOPIC", None)
            self.consume_topic = getenv("KAFKA_CONSUME_TOPIC", None)
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

    def serve(self):
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
            self._process_zmq()
        elif self.broker == Broker.kafka:
            self._process_kafka()

    def _process_zmq(self):
        try:
            while True:
                # Get frames
                data = self.consumer.receive(timeout=1)
                if data is None:
                    continue
                # Convert metadata to docarray
                frame_docs = DocumentArray.from_bytes(data)
                result = self.__call__(frame_docs)
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

    def _process_kafka(self):
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
                frame_docs = DocumentArray.from_bytes(data)
                result = self.__call__(frame_docs)
                # Process Results
                if self.produce_topic:
                    self.producer.produce(self.produce_topic, value=result.to_bytes())
        except (SystemExit, KeyboardInterrupt):
            print("Exit due to keyboard interrupt")
        except Exception as ex:
            print("Python error with no Exception handler:")
            print("Traceback error:", ex)
            traceback.print_exc()
        finally:
            self.consumer.close()
