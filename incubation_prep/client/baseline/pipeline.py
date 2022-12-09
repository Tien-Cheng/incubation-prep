from typing import Dict, Optional
from threading import Thread

from docarray import Document
from baseline.components import (
    Component,
    YOLODetector,
    ObjectTracker,
    SaveStream,
    StreamOutput,
)


class BaselinePipeline:
    def __init__(
        self,
        yolo_weights: str,
        embedder: str,
        output_address: str,
        output_port: str,
        zmq: bool,
        output_path: str,
        embedder_wts: Optional[str] = None,
        triton_url: Optional[str] = None
    ):
        self.components: Dict[str, Component] = {
            "det": YOLODetector(yolo_weights, "0", 640),
            "tracker": ObjectTracker(
                {
                    "embedder": embedder,
                    "embedder_model_name": "mobilenet",
                    "embedder_model_version": "1",
                    "embedder_wts": embedder_wts,
                    "triton_url": triton_url,
                }
            ),
            "output": StreamOutput(address=output_address, port=output_port, zmq=zmq),
            "save": SaveStream(path=output_path),
        }

    def __call__(self, frame: Document) -> Document:
        frame = self.components["det"]._call_main(frame)
        frame = self.components["tracker"]._call_main(frame)
        Thread(target=self.__display_frame, args=(frame, self.components["output"], self.components["save"]), daemon=True).start()
        return frame

    @staticmethod
    def __display_frame(frame, stream: StreamOutput, save: SaveStream):
        stream._call_main(frame)
        save._call_main(frame)

