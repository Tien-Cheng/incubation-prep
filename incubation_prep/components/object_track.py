from typing import List, Tuple

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


class ObjectTracker:
    def __init__(self, max_age: int = 30, nn_budget: int = 10):
        self.tracker = DeepSort(
            max_age=max_age,
            nn_budget=nn_budget,
        )

    def track(self, frame: np.ndarray, dets: List[Tuple]):
        tracks = self.tracker.update_tracks(frame=frame, raw_detections=dets)
        return tracks