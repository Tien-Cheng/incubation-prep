from os import getenv
from datetime import datetime

import cv2

from docarray import DocumentArray

from component import Component


class SaveStream(Component):
    """"""

    def __init__(self, width: int = 1280, height: int = 720, path: str = "/var/nfs"):
        super().__init__()
        self.width = width
        self.height = height
        self.path = path

    def __call__(self, docs: DocumentArray, **kwargs):
        """Read frames and save them in NFS or Redis

        :param docs: _description_
        :type docs: DocumentArray
        """
        with self.timer:
            for frame in docs:
                # Get stream name
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
                            (0, 0, 255),  # BGR
                        )
                cv2.resize(frame.tensor, (self.width, self.height))
                # Save
                filename = f"video-{frame.tags['output_stream']}-frame-{frame.tags['frame_id']}-{str(datetime.now())}.jpg"
                path = f"{self.path}/{filename}"
                cv2.imwrite(str(path), frame.tensor)
            return docs


if __name__ == "__main__":
    executor = SaveStream(
        path=getenv("SAVE_DIR", "videos")
        width=int(getenv("OUTPUT_WIDTH", 1280)),
        height=int(getenv("OUTPUT_HEIGHT", 720)),
    )
    executor.serve()
