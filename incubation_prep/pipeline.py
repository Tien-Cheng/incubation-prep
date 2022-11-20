import asyncio
import time

from components.object_det import ObjectDetector
from components.object_track import ObjectTracker
from components.split_frames import VideoIO
from components.drawer import Drawer


class Pipeline:
    def __init__(self):
        self.video_io = VideoIO()
        self.detector = ObjectDetector()
        self.tracker = ObjectTracker()
        self.drawer = Drawer()

    def __call__(self, video_path: str, output_path: str):
        reader, fps, frame_size = self.video_io.init_reader(video_path)
        writer = self.video_io.init_writer(output_path, fps, frame_size)
        start = time.time()
        for frame_count, frame in self.video_io.split_frames(reader):
            dets = self.detector.detect([frame])
            tracks = self.tracker.track(frame, dets[0])
            asyncio.run(self.video_io.draw_frame(frame, tracks, writer, self.drawer))
        reader.release()
        writer.release()
        seconds_taken = time.time() - start
        print(f"Time taken: {seconds_taken} seconds")
        print(f"Avg FPS: {frame_count / seconds_taken}")
