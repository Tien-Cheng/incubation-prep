from io import BytesIO
from os import getenv
from typing import List, NamedTuple

import numpy as np
import uvicorn
from PIL import Image
from pydantic import BaseModel
from yolov5 import YOLOv5
from yolov5.models.common import Detections

from fastapi import FastAPI, File

app = FastAPI()
model = YOLOv5(getenv("weights", "yolov5s.pt"))


class BoundingBox(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int


class Detection(BaseModel):
    bbox: BoundingBox
    confidence: float
    class_name: str


@app.post("/infer")
def get_detections(frames: List[bytes] = File(), size: int = 640):
    frames = [np.array(Image.open(BytesIO(image))) for image in frames]
    preds: Detections = model.predict(frames, size)

    dets_per_image = [
        [
            Detection(
                bbox=BoundingBox(*det[:4].int().tolist()),
                confidence=det[4].item(),
                class_name=int(det[5].item()),
            )
            for det in image_dets
            if det.size()[0] != 0
        ]
        for image_dets in preds.pred
    ]

    return dets_per_image


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=4001)
