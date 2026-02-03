import os
import asyncio
import socketio
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from aiortc import RTCPeerConnection, RTCSessionDescription

from common.motion_camera_source import CameraSource
from motion_detection import MotionPredictor
from webrtc.motion_track import MotionVideoTrack

# 1. Define and Create Required Directories immediately
# This prevents the RuntimeError seen in your logs
ALERT_DIR = "motion_alert"
if not os.path.exists(ALERT_DIR):
    os.makedirs(ALERT_DIR)

sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
sio_app = socketio.ASGIApp(sio)

app = FastAPI()

app.mount("/socket.io", sio_app)
# Update mounting to use the correct 'motion_alert' folder
app.mount("/clips", StaticFiles(directory=ALERT_DIR), name="clips")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pcs = set()

camera = CameraSource(rtsp_url="rtsp://username:password@192.168.1.64/stream")
motion_engine = MotionPredictor()

@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join(BASE_DIR, "client", "index.html")
    return FileResponse(index_path)

@app.post("/offer")
async def offer(params: dict = Body(...)):
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)

    pc.addTrack(MotionVideoTrack(camera, motion_engine, sio=sio))
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

@app.on_event("shutdown")
async def on_shutdown():
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    camera.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)