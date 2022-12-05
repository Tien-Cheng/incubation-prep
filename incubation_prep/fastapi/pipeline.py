import asyncio
import datetime
import io
import json
import math
import threading
import time
from itertools import count
from time import perf_counter

import click
import numpy as np
import requests
from vidgear.gears import VideoGear


async def frame_pipe(frame, skip: bool = False):
    # Send frames to YOLODetector
    dets = None
    if skip:
        res = requests.post(
            "http://localhost:4001/infer",
            files={"frames": (None, io.BytesIO(frame["tensor"]))},
        )
        dets = res.json()
        dets = json.dumps(dets)
        # Send frame and det to Object Tracker
        res = requests.post(
            f"http://localhost:4002/infer/{frame['tags']['video_path']}",
            files={"frames": (None, io.BytesIO(frame["tensor"])), "dets": dets},
        )
        dets = res.json()
        dets = json.dumps(dets)
    # Output to stream
    res = requests.post(
        f"http://localhost:4003/{frame['tags']['video_path']}",
        files={"frames": (None, io.BytesIO(frame["tensor"])), "dets": dets},
    )

    # Save dets to db


def fire_and_forget(coro):
    global _loop
    if _loop is None:
        _loop = asyncio.new_event_loop()
        threading.Thread(target=_loop.run_forever, daemon=True).start()
    _loop.call_soon_threadsafe(asyncio.create_task, coro)


class Reconnecting_VideoGear:
    def __init__(self, cam_address, stabilize=False, reset_attempts=50, reset_delay=5):
        self.cam_address = cam_address
        self.stabilize = stabilize
        self.reset_attempts = reset_attempts
        self.reset_delay = reset_delay
        self.source = VideoGear(
            source=self.cam_address,
            stabilize=self.stabilize,
        ).start()
        self.fps = self.source.framerate
        self.running = True

    def read(self):
        if self.source is None:
            return None
        if self.running and self.reset_attempts > 0:
            frame = self.source.read()
            if frame is None:
                self.source.stop()
                self.reset_attempts -= 1
                print(
                    "Re-connection Attempt-{} occured at time:{}".format(
                        str(self.reset_attempts),
                        datetime.datetime.now().strftime("%m-%d-%Y %I:%M:%S%p"),
                    )
                )
                time.sleep(self.reset_delay)
                self.source = VideoGear(
                    source=self.cam_address, stabilize=self.stabilize
                ).start()
                self.fps = self.source.framerate
                # return previous frame
                return self.frame
            else:
                self.frame = frame
                return frame
        else:
            return None

    def stop(self):
        self.running = False
        self.reset_attempts = 0
        self.frame = None
        if not self.source is None:
            self.source.stop()


class Pipeline:
    def __init__(self):
        self.streams = {}

    @staticmethod
    def read_video(cap: Reconnecting_VideoGear, path: str, output_stream: str):
        try:
            for frame_id in count():
                frame = cap.read()
                if frame is None:
                    break
                yield {
                    "id": f"{path}-frame-{frame_id}",
                    "tensor": np.array(frame),
                    "tags": {
                        "video_path": path,
                        "frame_id": frame_id,
                        "output_stream": output_stream,
                    },
                }
        finally:
            cap.release()

    def __call__(
        self,
        video_path: str,
        output_path: str,
        infer_fps: int = 1,
    ):
        start = perf_counter()
        frames = 0
        cap = Reconnecting_VideoGear(video_path)
        fps = 25 if math.isinf(cap.fps) else cap.fps
        infer_frame_skip = round(fps / infer_fps)
        try:
            for frame in self.read_video(cap, video_path, output_path):
                fire_and_forget(frame_pipe(frame, frames % infer_frame_skip == 0))
                frames += 1
        finally:
            end = perf_counter()
            mean_fps = frames / (end - start)
            print(f"Average FPS: {mean_fps}")


@click.command()
@click.option("-v", "--video")
@click.option("-o", "--output", default="Test")
@click.option("--infer-fps", default=4)
@click.option("--no-track", is_flag=True)
def main(video: str, output: str, infer_fps: int, no_track: bool):
    pipe = Pipeline()
    pipe(video, output, infer_fps, no_track)


if __name__ == "__main__":
    main()
