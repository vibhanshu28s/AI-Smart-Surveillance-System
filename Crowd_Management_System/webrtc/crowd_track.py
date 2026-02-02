import cv2
import asyncio
from av import VideoFrame
from aiortc import VideoStreamTrack

class CrowdVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, camera, crowd_model, sio=None):
        super().__init__()
        self.camera = camera
        self.crowd_model = crowd_model  # This is the CrowdManager class
        self.sio = sio

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = self.camera.read()

        if frame is None:
            # Maintain ~30fps if frame is dropped
            await asyncio.sleep(0.03)
            return await self.recv()

        # 1. Run the Crowd Management logic
        # Expects: processed_frame, is_overcrowded
        processed, is_overcrowded = self.crowd_model.predict(frame)

        # 2. Emit Real-time Data to Frontend
        if self.sio:
            # We emit an alert only if overcrowded, but you could
            # also emit the live count here if needed.
            if is_overcrowded:
                asyncio.ensure_future(self.sio.emit('crowd_alert', {
                    'message': 'MAX CAPACITY REACHED',
                    'timestamp': pts
                }))

        # 3. Convert BGR (OpenCV) to RGB (aiortc)
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame