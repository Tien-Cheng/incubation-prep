import cv2
import uvicorn
from vidgear.gears import VideoGear

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect

app = FastAPI()


@app.websocket("/ws/{stream}")
async def get_stream(websocket: WebSocket, stream: str):
    await websocket.accept()
    try:
        cam = VideoGear(source=f"rtsp://localhost:8554/{stream}")
        while True:
            frame = cam.read()
            if frame is None:
                break

            ret, buffer = cv2.imencode(".jpg", frame)
            await websocket.send_bytes(buffer.tobytes())
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        cam.stop()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=4003)
