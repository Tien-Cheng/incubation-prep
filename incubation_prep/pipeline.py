import asyncio
import time
import datetime
from typing import List, Optional

from components.drawer import Drawer
from components.object_det import ObjectDetector
from components.object_track import ObjectTracker
from components.split_frames import VideoIO


class Pipeline:
    def __init__(
        self,
        infer_fps: int = 4,
        seconds_between_frames: int = 1,
        classes: Optional[List[str]] = None,
    ):
        if classes is None:
            classes = ["person"]
        self.video_io = VideoIO(infer_fps, seconds_between_frames)
        self.detector = ObjectDetector(classes=classes)
        self.tracker = ObjectTracker()
        self.drawer = Drawer()

    def __call__(self, video_path: str, output_path: str):
        reader, fps, frame_size = self.video_io.init_reader(video_path)
        writer = self.video_io.init_writer(output_path, fps, frame_size)
        start = time.time()

        infer_frame_skip = round(fps / self.video_io.fps)
        for frame_count, frame in self.video_io.split_frames(reader):
            if (frame_count % infer_frame_skip) != 0:
                continue
            dets = self.detector.detect([frame])
            tracks = self.tracker.track(frame, dets[0])
            asyncio.run(self.video_io.draw_frame(frame, tracks, writer, self.drawer))
        reader.release()
        writer.release()
        seconds_taken = time.time() - start
        time_taken = datetime.timedelta(seconds=seconds_taken)
        print(f"Time taken for {video_path}: {time_taken}")
        print(f"Avg FPS: {frame_count / seconds_taken}")