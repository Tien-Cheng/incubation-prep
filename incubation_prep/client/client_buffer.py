import asyncio
import gc  # Prevent memory leak through manual gc
import threading
import multiprocessing as mp
from enum import Enum
from queue import Queue
from itertools import count
from os import getenv
from io import BytesIO
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
        self.start_frame_times = {}  # map frame id to start frame time

        self.metrics_producer = Producer(
            {
                "bootstrap.servers": getenv("KAFKA_ADDRESS", "127.0.0.1:9092"),
            }
        )

    def run(
        self,
        broker: BrokerType,
        buffer: mp.Queue,
        encode_queue: mp.Queue,
        cap: cv2.VideoCapture,
        video_path: str,
        output_stream: str,
        output_path: Optional[str] = None,
        nfs: bool = False,
        redis: bool = False,
        encode_numpy: bool = False,
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
            print("Starting to fill buffer. This should be seen only once.")
            send_proc = mp.Process(
                target=self.infer,
                args=(
                    buffer,
                    broker,
                ),
            )
            encode_process = mp.Process(
                target=self.encode_frames,
                args=(encode_queue,),
            )
            for frame_count in count():
                start = perf_counter()
                success, frame = cap.read()
                # print("Time to read frame", perf_counter() - start)
                frame_id = f"vid-{filename}-{output_stream}-frame-{frame_count}"
                if not success:
                    print("Error reading frame")
                # start = perf_counter()
                doc = Document(
                    id=frame_id,
                    tags={
                        "frame_id": frame_count,
                        "video_path": video_path,
                        "output_stream": output_stream,
                    },
                )
                doc.tags["numpy"] = encode_numpy
                doc.tags["shape"] = list(frame.shape)
                # print("Time to create doc", perf_counter() - start)
                # start = perf_counter()
                if nfs or redis:
                    # Call encode in a separate process
                    if nfs:
                        path = f"{output_path}/{frame_id}"
                        if encode_numpy:
                            path += ".npy"
                        else:
                            path += ".jpg"
                        doc.uri = path
                        doc.tags["transfer_mode"] = "nfs"
                    elif redis:
                        doc.tags["transfer_mode"] = "redis"
                        doc.tags["redis"] = frame_id
                    # Pass to encode queue
                    encode_queue.put((doc, frame))
                else:
                    # Otherwise, still need encode here
                    if encode_numpy:
                        doc.tensor = frame
                    else:
                        frame = cv2.imencode(".jpg", frame)[1].tobytes()
                        doc.blob = frame
                # print("Time to encode", perf_counter() - start)
                self.metrics_producer.produce(
                    "metrics",
                    value=dumps(
                        {
                            "type": "client_fill_buffer_time",
                            "timestamp": datetime.now().isoformat(),
                            "executor": "client",
                            "executor_id": "client",
                            "value": perf_counter() - start,
                        }
                    ).encode("utf-8"),
                )
                self.metrics_producer.poll(0)
                buffer.put(doc)
                if frame_count == 1:
                    encode_process.start()
                if frame_count == 100:
                    send_proc.start()
        except Exception as e:
            print("Error in populate frame buffer", e)
            pass
        finally:
            send_proc.kill()
            encode_process.kill()

    def encode_frames(self, encode_queue: mp.Queue):
        while True:
            doc, frame = encode_queue.get(block=True, timeout=5)
            mode = doc.tags["transfer_mode"]
            encode_numpy = doc.tags["numpy"]
            if mode == "nfs":
                path = doc.uri
                if encode_numpy:
                    np.save(path, np.array(frame))
                else:
                    cv2.imwrite(path, frame)
            elif mode == "redis":
                if encode_numpy:
                    np_bytes = BytesIO()
                    np.save(np_bytes, frame, allow_pickle=True)
                    frame = np_bytes.getvalue()
                else:
                    frame = cv2.imencode(".jpg", frame)[1].tobytes()
                frame_id = doc.tags["redis"]
                self.rds.set(frame_id, frame)
            else:
                return

    def read_frames(
        self,
        buffer: mp.Queue,
    ):
        try:
            while True:
                start = perf_counter()
                # Get doc from buffer
                doc = buffer.get(block=True, timeout=5)
                # start = perf_counter()
                # print("Time to encode", perf_counter() - start)
                self.metrics_producer.produce(
                    "metrics",
                    value=dumps(
                        {
                            "type": "client_yield_time",
                            "timestamp": datetime.now().isoformat(),
                            "executor": "client",
                            "executor_id": "client",
                            "value": perf_counter() - start,
                        }
                    ).encode("utf-8"),
                )
                yield doc
                sleep(0.5)
                start = perf_counter()
                self.metrics_producer.produce(
                    "metrics",
                    value=dumps(
                        {
                            "type": "start_processing",
                            "event": "overall",
                            "timestamp": datetime.now().isoformat(),
                            "executor": "client",
                            "executor_id": "client",
                        }
                    ).encode("utf-8"),
                )
                self.metrics_producer.poll(0)
                self.metrics_producer.produce(
                    "metrics",
                    value=dumps(
                        {
                            "type": "client_metric_produce_time",
                            "timestamp": datetime.now().isoformat(),
                            "executor": "client",
                            "executor_id": "client",
                            "value": perf_counter() - start,
                        }
                    ).encode("utf-8"),
                )
                self.metrics_producer.poll(0)
        except Exception as e:
            print("End generator")
            print(e)

    @staticmethod
    async def send_async_jina(frame: Document, client: JinaClient):
        async for _ in client.post(
            on="/infer", inputs=frame, request_size=1, return_responses=True
        ):
            continue

    def send_async_kafka(self, frame: Document, producer: Producer, topic: str):
        start = perf_counter()
        serialized_docarray = DocumentArray([frame]).to_bytes()
        # print("Time to serialize", perf_counter() - start)
        # TODO: Add a callback here, log the time, msg
        # we want to find out if frames are being sent
        # properly to Kafka
        # so ensure FPS is as expected, and how many
        # frames are being sent.

        self.start_frame_times[frame.tags["frame_id"]] = start
        producer.produce(
            topic,
            value=serialized_docarray,
            on_delivery=lambda err, msg: self.on_message(err, msg),
        )
        producer.poll(0)

    @staticmethod
    async def send_async_zmq(frame: Document, socket: zmq.asyncio.Socket):
        serialized_docarray = DocumentArray([frame]).to_bytes()
        await socket.send_multipart([serialized_docarray])

    @staticmethod
    def send_sync_baseline(frame: Document, pipe: BaselinePipeline):
        pipe(DocumentArray([frame]))

    def on_message(self, err: str, message: Message):
        time = perf_counter()
        data = DocumentArray.from_bytes(message.value())
        frame_id = data[0].tags["frame_id"]

        print(f"Time to receive frame: {time - self.start_frame_times[frame_id]}")
        fps = 1 / (time - self.start_frame_times[frame_id])
        # self.previous_frame_time = time
        print(f"FPS: {fps}, Expected FPS: {self.video_fps}, Error: {err}")

    def infer(
        self,
        buffer: mp.Queue,
        broker: BrokerType,
    ):
        print("Starting inference...")
        if broker == BrokerType.kafka:
            self.kafka_client.flush()
        if broker == BrokerType.jina:
            generator = self.read_frames(buffer)
            self.jina_client.post(
                on="/infer",
                inputs=generator,
                stream=True,
                request_size=1,
                return_responses=True,
            )
        else:
            try:
                for frame in self.read_frames(buffer):
                    if broker == BrokerType.jina:
                        fire_and_forget(self.send_async_jina(frame, self.jina_client))
                        # gc.collect()
                    elif broker == BrokerType.kafka:
                        self.send_async_kafka(
                            frame, self.kafka_client, self.producer_topic
                        )
                    elif broker == BrokerType.zmq:
                        fire_and_forget(self.send_async_zmq(frame, self.zmq_client))
                        # gc.collect()
                    elif broker == BrokerType.baseline:
                        self.send_sync_baseline(frame, self.baseline_pipe)
                    else:
                        raise NotImplementedError
            finally:
                if broker == BrokerType.kafka:
                    self.kafka_client.flush()


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
@click.option("--numpy-encode", is_flag=True)
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
            "host": getenv("JINA_HOSTNAME", "10.101.205.251"), # 
            "port": int(getenv("JINA_PORT", 4091)),
            "tracing": True,
            "traces_exporter_host": "192.168.168.107",
            "traces_exporter_port": 4317,
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
    try:
        buffer = mp.Queue()
        encode_queue = mp.Queue()
        cap = cv2.VideoCapture(video)
        client.run(
            broker,
            buffer,
            encode_queue,
            cap,
            video,
            stream_name,
            output_path,
            nfs,
            redis,
            encode_numpy,
        )
    except Exception as e:
        print("ERROR")
        print(e)
    finally:
        print("End")
        cap.release()


if __name__ == "__main__":
    main()
