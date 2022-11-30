import numpy as np
from time import sleep
import cv2
from typing import Dict
from threading import Thread
from jina import DocumentArray, Document, Executor, requests

def write_frame(frame: Document, stream: cv2.VideoWriter):
    if frame.matches:
        print("Drawing BBOX")
        bboxes, scores, classes = frame.matches[
            :, ("tags__bbox", "tags__score", "tags__class_name")
        ]
        for (bbox, score, class_) in zip(bboxes, scores, classes):
            l, t, r, b = tuple(map(int, bbox))
            cv2.rectangle(frame.tensor, (l, t), (r, b), (255, 255, 0), 1)
            cv2.putText(
                frame.tensor,
                f"{class_} ({score * 100:.2f}%)",
                (l, t - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
            )
    stream.write(frame.tensor)

class StreamOutput(Executor):
    """"""

    def __init__(
        self,
        pipeline: str = "appsrc ! videoconvert ! x264enc speed-preset=ultrafast bitrate=600 key-int-max=40 ! rtspclientsink location={}/{}",
        url: str = "rtsp://localhost:8554",
        fps: int = 30,
        width: int = 1280,
        height: int = 720,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.url = url
        self.pipeline = pipeline
        self.fps = fps
        self.width = width
        self.height = height
        self.streams: Dict[str, cv2.VideoWriter] = {}

    def create_stream(self, name: str):
        pipe = self.pipeline.format(self.url, name)
        self.logger.info(f"Creating stream for: {pipe}")
        self.streams[name] = cv2.VideoWriter(
            pipe,
            cv2.CAP_GSTREAMER,
            0,
            self.fps,
            (self.width, self.height),
            True,
        )

    @requests
    def produce(self, docs: DocumentArray, **kwargs):
        """Read frames and send them to their respective output streams

        :param docs: _description_
        :type docs: DocumentArray
        """
        for frame in docs:
            # Get stream name
            output_stream: str = frame.tags["output_stream"]
            if output_stream not in self.streams:
                self.create_stream(output_stream)
            Thread(
                target=write_frame,
                args=(frame, self.streams[output_stream]),
                daemon=True,
            ).start()
