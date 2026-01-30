import os
import asyncio
import socketio  # Requires: pip install python-socketio
from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
from aiortc import RTCPeerConnection, RTCSessionDescription

from common.camera_source import CameraSource
from PPE_Detection.ppe_prediction import PPEPredictor
from webrtc.video_track import PPEVideoTrack

# --- SOCKET.IO SETUP ---
# Initialize the Socket.io async server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = FastAPI()
# Wrap FastAPI with Socket.io
socket_app = socketio.ASGIApp(sio, app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pcs = set()

camera = CameraSource("rtsp://username:password@192.168.1.64/stream")
ppe = PPEPredictor()

@app.get("/")
async def index():
    # Ensure index.html is served correctly
    return FileResponse(os.path.join(BASE_DIR, "../client", "index.html"))

@app.post("/offer")
async def offer(params: dict = Body(...)):
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    # Pass the 'sio' instance to your track to enable alerts
    pc.addTrack(PPEVideoTrack(camera, ppe, sio=sio))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

if __name__ == "__main__":
    import uvicorn
    # IMPORTANT: Run the socket_app wrapper, not the FastAPI app
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)