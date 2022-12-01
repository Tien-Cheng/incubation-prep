import click
import cv2
import numpy as np

from time import perf_counter, sleep
from itertools import count
from typing import Dict

from docarray import Document, DocumentArray

from components.component import Component
from components.det import YOLODetector
from components.track import ObjectTracker
from components.output import StreamOutput


class Pipeline:
    components: Dict[str, Component] = {
        "det": YOLODetector("grpc://172.20.0.4:8001", "0", 640),
        "tracker": ObjectTracker(),
        "output": StreamOutput(),
    }

    buffer_dets = DocumentArray()

    @staticmethod
    def read_video(path: str, output_stream: str):
        cap = cv2.VideoCapture(path)
        try:
            for frame_id in count():
                success, frame = cap.read()
                if not success:
                    break
                yield DocumentArray(
                    Document(
                        id=f"{path}-frame-{frame_id}",
                        tensor=np.asarray(frame),
                        tags={
                            "video_path": path,
                            "frame_id": frame_id,
                            "output_stream": output_stream,
                        },
                    )
                )
                sleep(1/60)
        finally:
            cap.release()

    def __call__(self, video_path: str, output_path: str, infer_frame_skip: int = 1, no_track: bool = False):
        start = perf_counter()
        frames = 0
        try:
            for frame in self.read_video(video_path, output_path):
                # skip every n frames
                if  frames % infer_frame_skip == 0:
                    frame = self.components["det"](frame)
                    if not no_track:
                        frame = self.components["tracker"](frame)
                    self.buffer_dets = DocumentArray(frame[0].matches, copy=True)
                else:
                    # if skip frame, just use previous frame prediction
                    if self.buffer_dets:
                        frame[0].matches = self.buffer_dets
                frame = self.components["output"](frame)
                frames += 1
        finally:
            end = perf_counter()
            mean_fps = frames / (end - start)
            print(f"Average FPS: {mean_fps}")


@click.command()
@click.option("-v", "--video")
@click.option("-o", "--output", default="Test")
@click.option("--skip-infer-frames", default=1)
@click.option("--no-track", is_flag=True)
def main(video: str, output: str, skip_infer_frames: int, no_track: bool):
    pipe = Pipeline()
    pipe(video, output, skip_infer_frames, no_track)


if __name__ == "__main__":
    main()
