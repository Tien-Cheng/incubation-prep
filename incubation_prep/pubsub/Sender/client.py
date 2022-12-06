from os import getenv
from itertools import count
from time import perf_counter, sleep
import datetime

import click
import cv2
import numpy as np

from docarray import DocumentArray, Document
from confluent_kafka import Producer

_loop = None


class KafkaClient:
    def __init__(
        self, bootstrap_servers: str = "localhost:9092", producer_topic: str = "frames"
    ):
        self.client = Producer(
            {
                "bootstrap.servers": bootstrap_servers,
            }
        )
        self.producer_topic = producer_topic

    @staticmethod
    def read_frames(cap: cv2.VideoCapture, video_path, fps: int = 30):
        try:
            for frame_count in count():
                success, frame = cap.read()
                yield Document(
                    tensor=np.array(frame),
                    tags={
                        "frame_id": frame_count,
                        "video_path": video_path,
                        "output_stream": "test2",
                    },
                )
        finally:
            cap.release()
        return

    def infer(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        for frame in self.read_frames(cap, video_path):
            serialized_docarray = DocumentArray([frame]).to_bytes()
            self.client.produce(self.producer_topic, value=serialized_docarray)
            print("Sent frame!")
            sleep(1 / 30)


@click.command()
@click.option("--video", "-v", type=click.Path(exists=True))
def main(video):
    client = KafkaClient(
        bootstrap_servers=getenv("KAFKA_ADDRESS", "127.0.0.1:9092"),
        producer_topic=getenv("KAFKA_PRODUCE_TOPIC", "frames"),
    )
    client.infer(video)


if __name__ == "__main__":
    main()
