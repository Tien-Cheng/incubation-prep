import asyncio
import gc  # Prevent memory leak through manual gc
import threading
import multiprocessing
from io import BytesIO
from enum import Enum
from itertools import count
from os import getenv
from pathlib import Path
from time import sleep, perf_counter
from typing import Dict, Optional
from json import dumps
from datetime import datetime

import redis
import click
import cv2
import numpy as np
import zmq
import zmq.asyncio

from confluent_kafka import Producer, Message
from docarray import Document, DocumentArray
from jina import Client as JinaClient

from baseline.pipeline import BaselinePipeline

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
        baseline_config: Optional[Dict] = None,
        redis_config: Optional[Dict] = None,
        producer_topic: str = "frames",
    ):
        self.jina_client = None
        self.kafka_client = None
        self.zmq_client = None
        self.rds = None
        if jina_config is not None:
            self.jina_client = JinaClient(asyncio=False, **jina_config)
            self.jina_config = jina_config
        if kafka_config is not None:
            self.kafka_client = Producer(kafka_config)
            self.kafka_config = kafka_config
        if zmq_config is not None:
            self.zmq_context = zmq.asyncio.Context()
            self.zmq_client = self.zmq_context.socket(zmq.PUB)
            self.zmq_client.bind(f"tcp://{zmq_config['host']}:{zmq_config['port']}")
            self.zmq_config = zmq_config
        if baseline_config is not None:
            self.baseline_config = baseline_config
            self.baseline_pipe = BaselinePipeline(**baseline_config)
        if redis_config is not None:
            self.rds = redis.Redis(**redis_config)
        self.producer_topic = producer_topic
        self.previous_frame_time = perf_counter()
        self.start_frame_times = {} # map frame id to start frame time

        self.metrics_producer = Producer({
            "bootstrap.servers": getenv("KAFKA_ADDRESS", "127.0.0.1:9092"),
        })

    def read_frames(
        self,
        cap: cv2.VideoCapture,
        video_path: str,
        output_stream: str,
        output_path: Optional[str] = None,
        send_tensor: bool = True,
        nfs: bool = False,
        redis: bool = False,
        encode_numpy: bool = False
    ):
        try:
            # Try to get FPS
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if np.isinf(fps):
                fps = 25
            self.video_fps = fps
            filename = Path(video_path).stem
            if output_path is not None:
                Path(output_path).mkdir(parents=True, exist_ok=True)
            print("Starting. This should be seen only once.")
            for frame_count in count():
                success, frame = cap.read()
                # print("Time to read frame", perf_counter() - start)
                frame_id = f"vid-{filename}-{output_stream}-frame-{frame_count}"
                if not success:
                    print("Error reading frame")
                if encode_numpy:
                    # Convert BGR to RGB
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
                # print("Time to create doc", perf_counter() - start)
                # start = perf_counter()
                if nfs:
                    path = f"{output_path}/{frame_id}"
                    if encode_numpy:
                        path += ".npy"
                        np.save(path, frame)
                    else:
                        path += ".jpg"
                        cv2.imwrite(path, frame)
        finally:
            print("Releasing video")
            cap.release()
        return

    def infer(
        self,
        broker: BrokerType,
        video_path: str,
        output_stream: str,
        output_path: Optional[str] = None,
        send_image: bool = True,
        redis_cache: bool = False,
        nfs_cache: bool = False,
        encode_numpy: bool = False,
    ):
        print("Starting caching...")
        cap = cv2.VideoCapture(video_path)
        self.read_frames(cap, video_path, output_stream, output_path, send_image, nfs_cache, redis_cache, encode_numpy)

@click.command()
@click.option("--broker", "-b", type=click.Choice(["jina", "kafka", "zmq", "baseline"]))
@click.option("--video", "-v", type=click.Path(exists=True))
@click.option("--stream-name", "-s", type=str)
@click.option(
    "--output-path",
    required=False,
    type=click.Path(),
    default=getenv("NFS_MOUNT", "/data"),
)
@click.option("--send-image/--no-send-image", default=True)
@click.option("--nfs", is_flag=True)
@click.option("--redis", is_flag=True)
@click.option("--encode-numpy", is_flag=True)
@click.option("--load-baseline", is_flag=True)
def main(
    broker: BrokerType,
    video: str,
    stream_name: str,
    output_path: str,
    send_image: bool,
    nfs: bool,
    redis: bool,
    encode_numpy: bool,
    load_baseline: bool,
):
    if redis and nfs:
        raise ValueError("Cannot have both redis and nfs enabled at same time")
    if not send_image and not output_path and nfs:
        raise ValueError("Don't know where NFS is!")
    client = Client(
        jina_config={
            "host": getenv("JINA_HOSTNAME", "0.0.0.0"), # 10.101.205.251
            "port": int(getenv("JINA_PORT", 4091)),
            "tracing" : True,
            "traces_exporter_host" : "192.168.168.107",
            "traces_exporter_port" : 4317
        },
        kafka_config={
            "bootstrap.servers": getenv("KAFKA_ADDRESS", "127.0.0.1:9092"),
            "message.max.bytes": 1000000000,
        },
        zmq_config={
            "host": getenv("ZMQ_HOSTNAME", "*"),
            "port": getenv("ZMQ_PORT_OUT", "5555"),
        },
        baseline_config={
            "yolo_weights": getenv("YOLO_WEIGHTS", "baseline/weights/yolov5s.pt"),
            "output_address": getenv("OUTPUT_ADDRESS", "rtsp://127.0.0.1"),
            "output_port": getenv("OUTPUT_PORT", 8554),
            "zmq": bool(getenv("OUTPUT_USE_ZMQ", False)),
            "output_path": getenv("SAVE_DIR", "./final"),
        }
        if load_baseline
        else None,
        redis_config={
            "host": getenv("REDIS_HOST", "localhost"),
            "port": int(getenv("REDIS_PORT", 6379)),
            "db": int(getenv("REDIS_DB", 0)),
        }
        if redis
        else None,
        producer_topic=getenv("KAFKA_PRODUCE_TOPIC", "frames"),
    )
    client.infer(broker, video, stream_name, output_path, send_image, redis, nfs, encode_numpy)


if __name__ == "__main__":
    main()
