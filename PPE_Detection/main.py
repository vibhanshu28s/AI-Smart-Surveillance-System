import os
import asyncio
from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from aiortc import RTCPeerConnection, RTCSessionDescription

# Importing your existing custom modules
from common.camera_source import CameraSource
from PPE_Detection.ppe_prediction import PPEPredictor
from webrtc.video_track import PPEVideoTrack

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global store for PeerConnections
pcs = set()

# Initialize your Camera and PPE model globally (or via app state)
camera = CameraSource("rtsp://username:password@192.168.1.64/stream")
ppe = PPEPredictor()


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join(BASE_DIR, "client", "index1.html")
    return FileResponse(index_path)


@app.post("/offer")
async def offer(params: dict = Body(...)):
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"WebRTC state: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)

    # Adding the PPE processed track to the connection
    pc.addTrack(
        PPEVideoTrack(camera, ppe)
    )

    # Handle the WebRTC Handshake
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }


@app.on_event("shutdown")
async def on_shutdown():
    # Close all active WebRTC connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    import uvicorn

    # Note: Using port 8000 as requested for FastAPI convention
    uvicorn.run(app, host="0.0.0.0", port=8000)