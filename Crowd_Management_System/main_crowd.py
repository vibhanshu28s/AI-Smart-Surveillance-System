import os
import asyncio
from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, FileResponse
from aiortc import RTCPeerConnection, RTCSessionDescription

# Importing the Crowd Management modules we just created
from common.crowd_camera_source import CameraSource
from crowd_management import CrowdManager
from webrtc.crowd_track import CrowdVideoTrack

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global store for PeerConnections
pcs = set()

# Initialize Camera and Crowd Manager globally
# Replace with your actual RTSP URL or set to None for webcam
camera = CameraSource(rtsp_url=None)
crowd_manager = CrowdManager()

@app.get("/", response_class=HTMLResponse)
async def index():
    # Serve the crowd monitoring dashboard
    index_path = os.path.join(BASE_DIR, "client", "index.html")
    return FileResponse(index_path)

@app.post("/offer")
async def offer(params: dict = Body(...)):
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"WebRTC State: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)

    # Add the Crowd Management processed track to the WebRTC connection
    # This now streams frames with person counts and density alerts
    pc.addTrack(
        CrowdVideoTrack(camera, crowd_manager)
    )

    # Standard WebRTC Handshake
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }

@app.on_event("shutdown")
async def on_shutdown():
    # Gracefully close all active crowd monitoring streams
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    camera.release()

if __name__ == "__main__":
    import uvicorn
    # Running the crowd management server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)