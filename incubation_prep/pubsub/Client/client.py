from itertools import count
from os import getenv
from time import sleep

import click
import cv2
import numpy as np
from confluent_kafka import Producer
from docarray import Document, DocumentArray

_loop = None


class KafkaClient:
    def __init__(
        self, bootstrap_servers: str = "localhost:9092", producer_topic: str = "frames"
    ):
        self.client = Producer(
            {
                "bootstrap.servers": bootstrap_servers,
                "message.max.bytes": 1000000000,
            }
        )
        self.producer_topic = producer_topic

    @staticmethod
    def read_frames(cap: cv2.VideoCapture, video_path):
        try:
            # Try to get FPS
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if np.isinf(fps):
                fps = 25
            for frame_count in count():
                success, frame = cap.read()
                if not success:
                    print("Error reading frame")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield Document(
                    tensor=np.array(frame),
                    tags={
                        "frame_id": frame_count,
                        "video_path": video_path,
                        "output_stream": "test2",
                    },
                )
                sleep(1 / fps)
        finally:
            cap.release()
        return

    def infer(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        for frame in self.read_frames(cap, video_path):
            serialized_docarray = DocumentArray([frame]).to_bytes()
            self.client.produce(self.producer_topic, value=serialized_docarray)
            self.client.poll(0)
            print("Sent frame!")


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
