from itertools import count

import cv2

from jina import Client, Document


class JinaClient:
    def __init__(self, host: str = "localhost"):
        self.client = Client(host=host)

    @staticmethod
    def render_resp(resp):
        for d in resp.docs:
            cv2.imshow("output", d.tensor)

    @staticmethod
    def split_frames(video_path: str):
        cap = cv2.VideoCapture(video_path)
        for frame_count in count():
            success, frame = cap.read()
            if not success:
                break
            yield Document(
                tensor=frame, tags={"frame_id": frame_count, "video_path": video_path}
            )

    def infer(self, video_path: str):
        self.client.post(
            on="/infer",
            inputs=self.split_frames(video_path),
            on_done=self.render_resp,
            request_size=1,
        )
