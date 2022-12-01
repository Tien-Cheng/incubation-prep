from itertools import count
from time import perf_counter, sleep
from memory_profiler import profile

import click
import cv2
import numpy as np

from jina import Client, Document


class JinaClient:
    def __init__(self, host: str = "0.0.0.0", port: int = 4091):
        self.client = Client(host=host, port=port)

    def create_stream(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

    @staticmethod
    def read_frames(cap: cv2.VideoCapture, stream_name):
        for frame_count in count():
            success, frame = cap.read()
            if not success:
                break
            sleep(1 / 30)
            yield Document(
                tensor=np.array(frame),
                tags={
                    "frame_id": frame_count,
                    "video_path": stream_name,
                    "output_stream": "test2",
                },
            )

    
    @profile
    def infer(self, video_path: str):
        for frame in self.read_frames(self.cap, video_path):
            Document.generator_from_webcam
            self.client.post(
                on="/infer",
                inputs=frame,
                request_size=1,
            )


@click.command()
@click.option("--video", "-v", type=click.Path(exists=True))
def main(video):
    client = JinaClient()
    client.infer(video)


if __name__ == "__main__":
    prev_frame_time = perf_counter()
    main()
