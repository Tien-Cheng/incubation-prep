from io import BytesIO
from json import loads
from typing import Dict, List, NamedTuple, Optional, Tuple, Type, Union

import numpy as np
import uvicorn
from deep_sort_realtime.deep_sort.track import Track
from deep_sort_realtime.deepsort_tracker import DeepSort
from embedder import DeepSORTEmbedder
from PIL import Image
from pydantic import BaseModel

from fastapi import APIRouter, FastAPI, File, Form


class BoundingBox(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int


class Detection(BaseModel):
    bbox: BoundingBox
    confidence: float
    class_name: str
    track_id: int


class ObjectTracker:
    def __init__(self, embedder_kwargs: Optional[Dict] = None):
        self.router = APIRouter()
        self.router.add_api_route("/infer/{stream}", self.track, methods=["POST"])
        if embedder_kwargs is None:
            embedder_kwargs = {}
        self.embedder: DeepSORTEmbedder = DeepSORTEmbedder(**embedder_kwargs)

        self.trackers: Dict[str, DeepSort] = {}

    def track(
        self, stream: str, frames: List[bytes] = File(), dets_per_image: str = Form()
    ):
        if stream not in self.trackers:
            self.trackers[stream] = DeepSort(embedder=None)

        frames = [np.array(Image.open(BytesIO(image))) for image in frames]

        dets_per_image: List[List[Detection]] = loads(dets_per_image)
        dets_per_image: List[List[Tuple[List[Union[int, float]], float, str]]] = [
            [(det["bbox"], det["confidence"], det["class_name"]) for det in dets]
            for dets in dets_per_image
        ]

        results_per_image = []
        for frame, dets in zip(frames, dets_per_image):

            # Generate embeddings
            embeds = self.embedder(frame=frame, dets=dets)
            try:
                # Get tracking results
                tracks: List[Track] = self.trackers[stream].update_tracks(
                    dets, embeds=embeds, frame=frame
                )
            except IndexError:
                tracks = []

            # Process tracking results
            results = []
            for track in tracks:
                bbox: np.ndarray = track.to_ltrb()
                bbox = bbox.astype(np.int32).tolist()
                conf = track.get_det_conf()
                cls = track.get_det_class()
                track_id = track.track_id

                results.append(
                    Detection(
                        bbox=bbox, confidence=conf, class_name=cls, track_id=track_id
                    )
                )
            results_per_image.append(results)
        return results_per_image


tracker = ObjectTracker()
app = FastAPI()
app.include_router(tracker.router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=4002)
