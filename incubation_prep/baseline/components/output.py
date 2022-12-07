from typing import Dict, Optional, Union

import cv2
from vidgear.gears import NetGear, WriteGear

from jina import DocumentArray

from .component import Component


class StreamOutput(Component):
    """"""

    def __init__(
        self,
        address: str = "rtsp://localhost",
        port: str = "8554",
        fps: int = 30,
        width: int = 1280,
        height: int = 720,
        zmq: bool = False,
        ffmpeg_config: Optional[Dict] = None,
        zmq_config: Optional[Dict] = None,
    ):
        self.address = address
        self.port = port
        self.fps = fps
        self.width = width
        self.height = height
        self.zmq = zmq
        if ffmpeg_config is None:
            ffmpeg_config = {
                "-preset:v": "ultrafast",
                "-tune": "zerolatency",
                "-f": "rtsp",
                "-rtsp_transport": "tcp",
            }
        self.ffmpeg_config = ffmpeg_config
        if zmq_config is None:
            zmq_config = {
                "jpeg_compression": True,
                "jpeg_compression_quality": 90,
                "jpeg_compression_fastdct": True,
                "jpeg_compression_fastupsample": True,
            }
        self.zmq_config = zmq_config
        self.streams: Dict[str, Union[NetGear, WriteGear]] = {}

        if zmq:
            self.create_stream = self.create_stream_zmq

    def create_stream(self, name: str):
        self.streams[name] = WriteGear(
            f"{self.address}:{self.port}/{name}",
            logging=True,
            compression_mode=True,
            **self.ffmpeg_config,
        )

    def create_stream_zmq(self, name):
        self.streams[name] = NetGear(
            address=self.address,
            port=self.port,
            logging=True,
            receive_mode=False,
            **self.zmq_config,
        )

    def __call__(self, docs: DocumentArray, **kwargs):
        """Read frames and send them to their respective output streams

        :param docs: _description_
        :type docs: DocumentArray
        """
        for frame in docs:
            # Get stream name
            output_stream: str = frame.tags["output_stream"]
            if output_stream not in self.streams:
                self.create_stream(output_stream)
            # VidGear will handle threading for us
            if frame.matches:
                bboxes, scores, classes, track_ids = frame.matches[
                    :,
                    (
                        "tags__bbox",
                        "tags__confidence",
                        "tags__class_name",
                        "tags__track_id",
                    ),
                ]
                for (bbox, score, class_, id_) in zip(
                    bboxes, scores, classes, track_ids
                ):
                    l, t, r, b = tuple(map(int, bbox))
                    cv2.rectangle(frame.tensor, (l, t), (r, b), (0, 0, 255), 2)
                    cv2.putText(
                        frame.tensor,
                        f"[ID: {id_}] {class_} ({score * 100:.2f}%)",
                        (l, t - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),  
                    )
            frame.tensor = cv2.resize(frame.tensor, (self.width, self.height))
            frame.tensor = cv2.cvtColor(frame.tensor, cv2.COLOR_RGB2BGR)
            try:
                if self.zmq:
                    self.streams[output_stream].send(frame.tensor)
                else:
                    self.streams[output_stream].write(frame.tensor)
            except:
                pass
        return docs
