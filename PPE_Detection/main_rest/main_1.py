from aiohttp import web
from common.camera_source import CameraSource
from PPE_Detection.ppe_prediction import PPEPredictor
from webrtc.video_track1 import PPEVideoTrack
from aiortc import RTCPeerConnection, RTCSessionDescription
import os

pcs = set()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

async def index(request):
    return web.FileResponse(
        os.path.join(BASE_DIR, "../client", "index1.html")
    )

async def offer(request):
    params = await request.json()
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("WebRTC state:", pc.connectionState)
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)

    pc.addTrack(
        PPEVideoTrack(
            request.app["camera"],
            request.app["ppe"]
        )
    )

    await pc.setRemoteDescription(
        RTCSessionDescription(
            sdp=params["sdp"],
            type=params["type"]
        )
    )

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

async def on_shutdown(app):
    for pc in pcs:
        await pc.close()
    pcs.clear()

def main():
    camera = CameraSource("rtsp://username:password@192.168.1.64/stream")
    ppe = PPEPredictor()

    app = web.Application()
    app["camera"] = camera
    app["ppe"] = ppe

    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(on_shutdown)

    web.run_app(app, port=8000)

if __name__ == "__main__":
    main()
