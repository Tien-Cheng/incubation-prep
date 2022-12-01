from itertools import count
from time import perf_counter, sleep
from memory_profiler import profile

import click
import cv2
import threading
import gc
import asyncio
import numpy as np

from jina import Client, Document

_loop = None

def fire_and_forget(coro):
    global _loop
    if _loop is None:
        _loop = asyncio.new_event_loop()
        threading.Thread(target=_loop.run_forever, daemon=True).start()
    _loop.call_soon_threadsafe(asyncio.create_task, coro)

class JinaClient:
    def __init__(self, host: str = "0.0.0.0", port: int = 4091, use_async=False):
        self.use_async = use_async
        self.client = Client(host=host, port=port, asyncio=use_async)

    @staticmethod
    def read_frames(video_path, fps: int = 30):
        cap = cv2.VideoCapture(video_path)
        try:
            for frame_count in count():
                success, frame = cap.read()
                if not success:
                    break
                yield Document(
                    tensor=np.array(frame),
                    tags={
                        "frame_id": frame_count,
                        "video_path": video_path,
                        "output_stream": "test2",
                    },
                )
        finally:
            cap.release()
        return

    @staticmethod
    @profile
    async def send_async(frame: Document, client: Client):
        # start = perf_counter()
        async for _ in client.post(on="/infer", inputs=frame, request_size=1, return_responses=True):
            # print(f"Time: {perf_counter() - start}s")
            continue
    
    @staticmethod
    @profile
    def send_sync(frame: Document, client: Client):
        start = perf_counter()
        client.post(on="/infer", inputs=frame, request_size=1, return_responses=True)
        print(f"Time: {perf_counter() - start}s")

    def infer(self, video_path: str):
        for frame in self.read_frames(video_path):
            if self.use_async:
                fire_and_forget(self.send_async(frame, self.client))
                gc.collect() 
            else:
                self.send_sync(frame, self.client)
            sleep(1/30)


@click.command()
@click.option("--video", "-v", type=click.Path(exists=True))
def main(video):
    client = JinaClient()
    client.infer(video)


if __name__ == "__main__":
    main()
