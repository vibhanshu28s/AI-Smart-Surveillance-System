import os
import asyncio
import socketio # Added
from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from aiortc import RTCPeerConnection, RTCSessionDescription

# Importing your existing custom modules
from common.camera_source import CameraSource
from PPE_Detection.ppe_prediction import PPEPredictor
from webrtc.video_track1 import PPEVideoTrack
#
# from fastapi.staticfiles import StaticFiles
#
# # Add this before your routes
# # Assuming alert.mp3 is inside a folder named 'webrtc' based on your screenshot
# app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "webrtc"), html=True), name="static")

# 1. Initialize Socket.IO AsyncServer
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = FastAPI()

# 2. Wrap FastAPI with Socket.IO ASGI application
socket_app = socketio.ASGIApp(sio, app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pcs = set()

# Initialize globally
camera = CameraSource("rtsp://username:password@192.168.1.64/stream")
ppe = PPEPredictor()

@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join(BASE_DIR, "../client", "index1.html")
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

    # 3. Pass the 'sio' instance to your PPEVideoTrack
    pc.addTrack(
        PPEVideoTrack(camera, ppe, sio=sio)
    )

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }

@app.on_event("shutdown")
async def on_shutdown():
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
    import uvicorn
    # 4. CRITICAL: Run 'socket_app', not 'app'
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)