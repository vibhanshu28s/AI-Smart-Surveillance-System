import cv2
import asyncio
from av import VideoFrame
from aiortc import VideoStreamTrack

class MotionVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, camera, motion_model, sio=None):
        super().__init__()
        self.camera = camera
        self.motion_model = motion_model
        self.sio = sio

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = self.camera.read()

        if frame is None:
            # Maintain ~30 FPS loop if frame is missing
            await asyncio.sleep(0.03)
            return await self.recv()

        # 1. Run your Motion Detection model
        # The model returns the annotated image and a boolean (True/False)
        processed, motion_detected = self.motion_model.predict(frame)

        # 2. Emit Motion Alert
        if motion_detected and self.sio:
            # Using ensure_future to prevent network latency from lagging the video feed
            asyncio.ensure_future(self.sio.emit('motion_alert', {
                'message': 'Motion Detected!',
                'timestamp': pts
            }))

        # 3. Prepare frame for WebRTC
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame