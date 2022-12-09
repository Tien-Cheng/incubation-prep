from os import getenv
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import numpy as np
from component import Component
from docarray import Document, DocumentArray
from bytetracker import BYTETracker
from simpletimer import StopwatchKafka


class ObjectTracker(Component):
    """"""
    def __init__(self, **kwargs):
        super().__init__()
        self.trackers: Dict[str, BYTETracker] = {}
        self.non_triton_timer = StopwatchKafka(
            bootstrap_servers=getenv("KAFKA_ADDRESS", "127.0.0.1:9092"),
            kafka_topic=self.metrics_topic,
            metadata={"executor": self.executor_name},
            kafka_parition=-1,
        )



    def __call__(self, docs: DocumentArray, **kwargs):
        all_dets = docs.map(self._get_dets)
        for frame, dets in zip(docs, all_dets):
            if not frame.matches:
                continue
            output_stream: str = frame.tags["output_stream"]
            if output_stream not in self.trackers:
                self._create_tracker(output_stream)
            tracks = self.trackers[output_stream].update(dets, None)
            # Update matches using tracks
            frame.matches = self._update_dets(tracks)
        return docs

    def _create_tracker(self, name: str):
        tracker = BYTETracker()
        self.trackers[name] = tracker

    @staticmethod
    def _get_dets(frame: Document) -> List[Tuple[List[Union[int, float]], float, str]]:
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
    )
    executor.serve()
