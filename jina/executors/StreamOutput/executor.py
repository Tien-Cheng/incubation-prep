from time import sleep
import cv2
from typing import Dict
from jina import DocumentArray, Executor, requests


class StreamOutput(Executor):
    """"""

    def __init__(
        self,
        pipeline: str = "appsrc ! videoconvert ! x264enc speed-preset=ultrafast bitrate=600 key-int-max=40 ! rtspclientsink location={}/{}",
        url: str = "rtsp://localhost:8554",
        fps: int = 30,
        width: int = 1920,
        height: int = 1080,
        **kwargs
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
            self.streams[output_stream].write(frame.tensor)
            self.logger.info('Sent frame to server')
