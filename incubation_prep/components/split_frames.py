import math
import asyncio
from itertools import count
from pathlib import Path
from typing import Tuple, Union

import cv2


class VideoIO:
    def __init__(self, infer_fps: int = 30, seconds: int = 1):
        self.infer_fps = infer_fps
        self.seconds = seconds  # no of seconds between each chip save

    def split_frames(self, cap):
        for frame_count in count():
            try:
                success, frame = cap.read()
                if not success:
                    break
                yield frame_count, frame
            except Exception as e:
                cap.release()
                raise e

    async def draw_frame(frame, tracks, writer, drawer):
        drawn_frame = drawer.draw_tracks(frame, tracks)
        writer.write(drawn_frame)

    def init_reader(self, video_path: Union[Path, str]):
        if isinstance(video_path, str):
            video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file {video_path} does not exist")

        cap = cv2.VideoCapture(str(video_path))

        # Get FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 25 if math.isinf(fps) else fps

        # Get Video Width and Height
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return cap, fps, (width, height)

    def init_writer(
        self, video_path: Union[Path, str], fps: int, frame_size: Tuple[int, int]
    ):
        if isinstance(video_path, str):
            video_path = Path(video_path)

        if not video_path.parent.exists():
            video_path.parent.mkdir(parents=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)
        return writer
