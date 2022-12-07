from io import BytesIO
from json import loads
from os import getenv
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import uvicorn
from PIL import Image
from vidgear.gears import NetGear, WriteGear

from fastapi import APIRouter, FastAPI, File, Form


class StreamOutput:
    """"""

    def __init__(
        self,
        address: str = "rtsp://0.0.0.0",
        port: str = "8554",
        fps: int = 30,
        width: int = 1280,
        height: int = 720,
        zmq: bool = False,
        ffmpeg_config: Optional[Dict] = None,
        zmq_config: Optional[Dict] = None,
    ):
        self.api = APIRouter()
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

        self.api.add_api_route("/{stream}", self.stream, methods=["POST"])

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

    def stream(
        self,
        stream: str,
        frames: List[bytes] = File(),
        dets_per_image: Optional[Union[str, List]] = Form(None),
    ):
        """Read frames and send them to their respective output streams

        :param docs: _description_
        :type docs: DocumentArray
        """
        if stream not in self.streams:
            self.create_stream(stream)

        if isinstance(frames[0], bytes):
            frames = [
                np.array(Image.open(BytesIO(image)))[..., ::-1] for image in frames
            ]

        if not dets_per_image:
            dets_per_image = [None] * len(frames)

        if dets_per_image and isinstance(dets_per_image, str):
            dets_per_image: List[List] = loads(dets_per_image)
            dets_per_image: List[
                List[Tuple[List[Union[int, float]], float, str, int]]
            ] = [
                [
                    (det["bbox"], det["confidence"], det["class_name"], det["track_id"])
                    for det in dets
                ]
                for dets in dets_per_image
            ]

        for frame, dets in zip(frames, dets_per_image):
            # Get stream name
            # VidGear will handle threading for us
            if dets:
                bboxes, scores, classes, track_ids = list(zip(*dets))
                for (bbox, score, class_, id_) in zip(
                    bboxes, scores, classes, track_ids
                ):
                    l, t, r, b = tuple(map(int, bbox))
                    cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        f"[ID: {id_}] {class_} ({score * 100:.2f}%)",
                        (l, t - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),  # BGR
                    )
            cv2.resize(frame, (self.width, self.height))
            try:
                if self.zmq:
                    self.streams[stream].send(frame)
                else:
                    self.streams[stream].write(frame)
            except:
                pass


output = StreamOutput(
    address=getenv("ADDRESS", "127.0.0.1"),
    port=getenv("STREAM_PORT", "5555"),
    zmq=getenv("USE_ZMQ", True),
)
app = FastAPI()
app.include_router(output.api)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=4003)
