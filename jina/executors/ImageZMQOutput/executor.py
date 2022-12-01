import numpy as np
from time import sleep
import imagezmq
import cv2
from typing import Dict
from threading import Thread
from jina import DocumentArray, Document, Executor, requests


def write_frame(
    name: str, frame: Document, stream: imagezmq.ImageSender, width: int, height: int
):
    if frame.matches:
        bboxes, scores, classes, track_ids = frame.matches[
            :, ("tags__bbox", "tags__confidence", "tags__class_name", "tags__track_id")
        ]
        for (bbox, score, class_, id_) in zip(bboxes, scores, classes, track_ids):
            l, t, r, b = tuple(map(int, bbox))
            cv2.rectangle(frame.tensor, (l, t), (r, b), (255, 255, 0), 1)
            cv2.putText(
                frame.tensor,
                f"[ID: {id_}] {class_} ({score * 100:.2f}%)",
                (l, t - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
            )
    cv2.resize(frame.tensor, (width, height))
    stream.send_image(name, frame.tensor)
    frame.pop('tensor')


class StreamOutput(Executor):
    """"""

    def __init__(
        self,
        url: str = "tcp://127.0.0.1:5555",
        width: int = 1280,
        height: int = 720,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.url = url
        self.height = height
        self.width = width
        self.streams: Dict[str, cv2.VideoWriter] = {}

    def create_stream(self, name: str):
        self.streams[name] = imagezmq.ImageSender(self.url)

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
                args=(
                    output_stream,
                    frame,
                    self.streams[output_stream],
                    self.width,
                    self.height,
                ),
                daemon=True,
            ).start()