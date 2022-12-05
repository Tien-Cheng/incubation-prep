from io import BytesIO
from typing import List, NamedTuple

import numpy as np
import uvicorn
from PIL import Image
from pydantic import BaseModel
from yolov5 import YOLOv5
from yolov5.models.common import Detections

from fastapi import FastAPI, File

app = FastAPI()
model = YOLOv5("grpc://172.0.0.4:8001")


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
                bbox=BoundingBox(*det[0, :4].int().tolist()),
                confidence=det[0, 4].item(),
                class_name=det[0, 5].item(),
            )
            for det in image_dets
        ]
        for image_dets in preds.tolist()
    ]

    return dets_per_image


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=4001)
