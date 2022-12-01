import numpy as np

from typing import Dict, Optional, List, Tuple, Union

from jina import DocumentArray, Document, Executor, requests
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.track import Track


class ObjectTracker(Executor):
    """"""

    def __init__(self, embedder: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder
        self.trackers: Dict[str, DeepSort] = {}

    @requests
    def track(self, docs: DocumentArray, **kwargs):
        all_dets = docs.map(self._get_dets)
        for frame, dets in zip(docs, all_dets):
            if not frame.matches:
                continue
            output_stream: str = frame.tags["output_stream"]
            if output_stream not in self.trackers:
                self._create_tracker(output_stream)
            tracker = self.trackers[output_stream]
            # embedding of each cropped det
            if self.embedder == "triton":
                embeds = self._generate_embeds_triton(tracker, frame.tensor, dets)
            else:
                # embedding of each detection
                # if does not exist, return None
                # when None, will use the included mobilenet model
                embeds = frame.matches.embeddings
            image = frame.tensor
            tracks = self.trackers[output_stream].update_tracks(
                dets, frame=image, embeds=embeds
            )

            # Update matches using tracks
            frame.matches = self._update_dets(tracks)

    def _create_tracker(self, name: str):
        tracker = DeepSort(embedder=self.embedder or "mobilenet")
        self.trackers[name] = tracker

    @staticmethod
    def _get_dets(frame: Document) -> List[Tuple[List[Union[int, float]], float, str]]:
        return [
            (det.tags["bbox"], det.tags["confidence"], det.tags["class_name"])
            for det in frame.matches
        ]

    @staticmethod
    def _generate_embeds_triton(
        tracker: DeepSort,
        frame: np.ndarray,
        dets: List[Tuple[List[Union[int, float]], float, str]],
    ):
        cropped_dets = tracker.crop_bb(frame, dets)
        # Perform batch inference of all crops
        raise NotImplementedError

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
