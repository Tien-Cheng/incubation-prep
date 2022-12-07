import json
from os import getenv
from typing import Dict, List, Optional, Tuple, Type, Union
from datetime import datetime

import numpy as np
from component import Component
from deep_sort_realtime.deep_sort.track import Track
from deep_sort_realtime.deepsort_tracker import DeepSort
from docarray import Document, DocumentArray
from embedder import DeepSORTEmbedder, Embedder


class ObjectTracker(Component):
    """"""

    def __init__(self, embedder_kwargs: Optional[Dict] = None, **kwargs):
        super().__init__()
        if embedder_kwargs is None:
            embedder_kwargs = {}
        self.embedder = DeepSORTEmbedder(**embedder_kwargs)
        self.trackers: Dict[str, DeepSort] = {}

    def __call__(self, docs: DocumentArray, **kwargs):
        all_dets = docs.map(self._get_dets)
        for frame, dets in zip(docs, all_dets):
            if not frame.matches:
                continue
            output_stream: str = frame.tags["output_stream"]
            if output_stream not in self.trackers:
                self._create_tracker(output_stream)
            # embedding of each cropped det
            image = frame.tensor
            if not frame.matches.embeddings:
                embeds = self.embedder(image, dets)
                if self.embedder.embedder != Embedder.triton:
                    no_inferences = len(dets)
                    metric = {
                        "type": "non_triton_inference",
                        "timestamp": datetime.now().isoformat(),
                        "executor": self.executor_name,
                        "executor_id": self.executor_id,
                        "output_stream": frame.tags["output_stream"],
                        "video_source": frame.tags["video_path"],
                        "frame_id": frame.tags["frame_id"],
                        "value": no_inferences,
                    }
                    # Produce metric
                    self.metric_producer.produce(
                        self.metrics_topic, value=json.dumps(metric).encode("utf-8")
                    )
                    self.metric_producer.poll(0)
            else:
                embeds = frame.matches.embeddings
            tracks = self.trackers[output_stream].update_tracks(dets, embeds=embeds)
            # Update matches using tracks
            frame.matches = self._update_dets(tracks)
        return docs

    def _create_tracker(self, name: str):
        tracker = DeepSort(embedder=None)
        self.trackers[name] = tracker

    @staticmethod
    def _get_dets(frame: Document) -> List[Tuple[List[Union[int, float]], float, str]]:
        return [
            (det.tags["bbox"], det.tags["confidence"], det.tags["class_name"])
            for det in frame.matches
        ]

    @staticmethod
    def _update_dets(tracks: List[Track]) -> DocumentArray:
        results = []
        for track in tracks:
            bbox: np.ndarray = track.to_ltrb()
            bbox = bbox.astype(np.int32).tolist()
            conf = track.get_det_conf()
            cls = track.get_det_class()
            track_id = track.track_id
            if not (
                bbox is not None
                and conf is not None
                and cls is not None
                and track_id is not None
            ):
                continue
            results.append(
                Document(
                    tags={
                        "bbox": bbox,
                        "confidence": conf,
                        "class_name": cls,
                        "track_id": track_id,
                    }
                )
            )
        return DocumentArray(results)


if __name__ == "__main__":
    executor = ObjectTracker(
        {
            "embedder": getenv("TRACKER_EMBEDDER", "mobilenet"),
            "embedder_model_name": getenv("TRACKER_EMBEDDER_MODEL_NAME", None),
            "embedder_wts": getenv("TRACKER_EMBEDDER_WTS", None),
            "embedder_model_version": getenv("TRACKER_EMBEDDER_MODEL_VERSION", None),
            "triton_url": getenv("TRACKER_TRITON_URL", None),
        }
    )
    executor.serve()
