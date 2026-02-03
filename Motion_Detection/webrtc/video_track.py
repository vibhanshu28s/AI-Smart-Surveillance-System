import cv2
import asyncio
from av import VideoFrame
from aiortc import VideoStreamTrack

class PPEVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, camera, ppe_model, sio=None): # Added sio parameter
        super().__init__()
        self.camera = camera
        self.ppe_model = ppe_model
        self.sio = sio

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = self.camera.read()

        if frame is None:
            await asyncio.sleep(0.03)
            return await self.recv()

        # 3. Run your PPE model
        processed, violation = self.ppe_model.predict(frame)

        # --- NEW: Emit Warning ---
        if violation and self.sio:
            # We use ensure_future so the alert doesn't lag the video stream
            asyncio.ensure_future(self.sio.emit('ppe_violation_alert', {
                'message': 'PPE Violation Detected!',
                'type': violation.get('type', 'Unknown'),
                'timestamp': pts
            }))
        # -------------------------

        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame