import asyncio
import gc  # Prevent memory leak through manual gc
import threading

from os import getenv
from itertools import count
from time import sleep
from typing import Optional, Dict
from pathlib import Path
from enum import Enum

import cv2
import click
import numpy as np
import zmq, zmq.asyncio

from confluent_kafka import Producer
from docarray import Document, DocumentArray
from jina import Client as JinaClient

_loop = None


def fire_and_forget(coro):
    global _loop
    if _loop is None:
        _loop = asyncio.new_event_loop()
        threading.Thread(target=_loop.run_forever, daemon=True).start()
    _loop.call_soon_threadsafe(asyncio.create_task, coro)


class BrokerType(str, Enum):
    kafka = "kafka"
    zmq = "zmq"
    jina = "jina"
    baseline = "baseline"


class Client:
    def __init__(
        self,
        jina_config: Optional[Dict] = None,
        kafka_config: Optional[Dict] = None,
        zmq_config: Optional[Dict] = None,
        producer_topic: str = "frames",
    ):
        self.jina_client = None
        self.kafka_client = None
        self.zmq_client = None
        if jina_config is not None:
            self.jina_client = JinaClient(**jina_config)
            self.jina_config = jina_config
        if kafka_config is not None:
            self.kafka_client = Producer(kafka_config)
            self.kafka_config = kafka_config
        if zmq_config is not None:
            self.zmq_context = zmq.asyncio.Context()
            self.zmq_client = self.zmq_context.socket(zmq.PUB)
            self.zmq_client.bind(f"tcp://{zmq_config['host']}:{zmq_config['port']}")
            self.zmq_config = zmq_config
        self.producer_topic = producer_topic

    @staticmethod
    def read_frames(
        cap: cv2.VideoCapture,
        video_path: str,
        output_stream: str,
        output_path: Optional[str] = None,
        send_tensor: bool = True,
        nfs: bool = False,
        redis: bool = False,
    ):
        try:
            # Try to get FPS
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if np.isinf(fps):
                fps = 25
            filename = Path(video_path).stem
            for frame_count in count():
                success, frame = cap.read()
                frame_id = f"vid-{filename}-{output_stream}-frame-{frame_count}"
                if not success:
                    print("Error reading frame")
                doc = Document(
                    id=frame_id,
                    tags={
                        "frame_id": frame_count,
                        "video_path": video_path,
                        "output_stream": output_stream,
                    },
                )
                if not send_tensor:
                    # Save to NFS or Redis, and put URI in document
                    # Save to NFS
                    if nfs:
                        path = f"{output_path}/{frame_id}.jpg"
                        cv2.imwrite(path, frame)
                        doc.uri = path
                    elif redis:
                        raise NotImplementedError
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    doc.tensor = np.array(frame)
                yield doc
                sleep(1 / fps)
        finally:
            cap.release()
        return

    @staticmethod
    async def send_async_jina(frame: Document, client: JinaClient):
        async for _ in client.post(
            on="/infer", inputs=frame, request_size=1, return_responses=True
        ):
            continue

    @staticmethod
    def send_async_kafka(frame: Document, producer: Producer, topic: str):
        serialized_docarray = DocumentArray([frame]).to_bytes()
        producer.produce(topic, value=serialized_docarray)
        producer.poll(0)

    @staticmethod
    async def send_async_zmq(frame: Document, socket: zmq.asyncio.Socket):
        serialized_docarray = DocumentArray([frame]).to_bytes()
        await socket.send_multipart([serialized_docarray])

    def infer(
        self,
        broker: BrokerType,
        video_path: str,
        output_stream: str,
        output_path: Optional[str] = None,
        send_image: bool = True,
        redis_cache: bool = False,
        nfs_cache: bool = False,
    ):
        print("Starting inference...")
        cap = cv2.VideoCapture(video_path)
        for frame in self.read_frames(
            cap,
            video_path,
            output_stream,
            output_path,
            send_image,
            nfs_cache,
            redis_cache,
        ):
            if broker == BrokerType.jina:
                fire_and_forget(self.send_async_jina(frame, self.jina_client))
                gc.collect()
            elif broker == BrokerType.kafka:
                self.send_async_kafka(frame, self.kafka_client, self.producer_topic)
            elif broker == BrokerType.zmq:
                fire_and_forget(self.send_async_zmq(frame, self.zmq_client))
                gc.collect()
            else:
                raise NotImplementedError


@click.command()
@click.option("--broker", "-b", type=click.Choice(["jina", "kafka", "zmq", "baseline"]))
@click.option("--video", "-v", type=click.Path(exists=True))
@click.option("--stream-name", "-s", type=str)
@click.option("--output-path", required=False, type=click.Path())
@click.option("--send-image/--no-send-image", default=True)
@click.option("--nfs", is_flag=True)
@click.option("--redis", is_flag=True)
def main(
    broker: BrokerType,
    video: str,
    stream_name: str,
    output_path: str,
    send_image: bool,
    nfs: bool,
    redis: bool,
):
    if redis and nfs:
        raise ValueError("Cannot have both redis and nfs enabled at same time")
    if not send_image and not output_path and nfs:
        raise ValueError("Don't know where NFS is!")

    client = Client(
        jina_config={
            "host": getenv("JINA_HOSTNAME", "0.0.0.0"),
            "port": int(getenv("JINA_PORT", 4091)),
        },
        kafka_config={
            "bootstrap.servers": getenv("KAFKA_ADDRESS", "127.0.0.1:9092"),
            "message.max.bytes": 1000000000,
        },
        zmq_config={
            "host": getenv("ZMQ_HOSTNAME", "*"),
            "port": getenv("ZMQ_PORT_OUT", "5555"),
        },
        producer_topic=getenv("KAFKA_PRODUCE_TOPIC", "frames"),
    )
    client.infer(broker, video, stream_name, output_path, send_image, redis, nfs)

if __name__ == "__main__":
    main()