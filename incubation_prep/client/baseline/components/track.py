import numpy as np
import torch
from typing import Dict, List, Tuple, Union

from .component import Component
from docarray import Document, DocumentArray
from bytetracker import BYTETracker


class ObjectTracker(Component):
    """"""

    def __init__(self, name: str = "track-baseline",**kwargs):
        super().__init__(name=name)
        self.trackers: Dict[str, BYTETracker] = {}

    def __call__(self, docs: DocumentArray, **kwargs):
        all_dets = docs.map(self._get_dets)
        for frame, dets in zip(docs, all_dets):
            if not frame.matches:
                continue
            output_stream: str = frame.tags["output_stream"]
            if output_stream not in self.trackers:
                self._create_tracker(output_stream)
            # embedding of each cropped det
            tracks = self.trackers[output_stream].update(dets, None)
            # Update matches using tracks
            frame.matches = self._update_dets(tracks)
        return docs

    def _create_tracker(self, name: str):
        tracker = BYTETracker()
        self.trackers[name] = tracker

    @staticmethod
    def _get_dets(frame: Document) -> List[Tuple[List[Union[int, float]], float, str]]:
        # DeepSORT wants LTWH format instead of YOLO LTRB format
        det = [
            [
                *det.tags["bbox"],
                det.tags["confidence"],
                det.tags["class_name"]
            ]
            for det in frame.matches
        ]
        return torch.tensor(np.array(det, dtype=np.float32))

    @staticmethod
    def _update_dets(tracks: np.ndarray) -> DocumentArray:
        results = []
        try:
            tracks = list(tracks)
            for track in tracks:
                results.append(
                    Document(
                        tags={
                            "bbox": track[:4],
                            "confidence": track[6],
                            "class_name": track[5],
                            "track_id": track[4],
                        }
                    )
                )
        except Exception as e:
            print(e)
        return DocumentArray(results)


if __name__ == "__main__":
    executor = ObjectTracker(
        # {
        #     "embedder": getenv("TRACKER_EMBEDDER", "mobilenet"),
        #     "embedder_model_name": getenv("TRACKER_EMBEDDER_MODEL_NAME", None),
        #     "embedder_wts": getenv("TRACKER_EMBEDDER_WTS", None),
        #     "embedder_model_version": getenv("TRACKER_EMBEDDER_MODEL_VERSION", None),
        #     "triton_url": getenv("TRACKER_TRITON_URL", None),
        # }
    )
    executor.serve()
