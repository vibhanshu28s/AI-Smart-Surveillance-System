# import cv2
# import asyncio
# from av import VideoFrame
# from aiortc import VideoStreamTrack
#
# class PPEVideoTrack(VideoStreamTrack):
#     def __init__(self, camera, ppe_model):
#         super().__init__()
#         self.camera = camera
#         self.ppe_model = ppe_model
#
#     async def recv(self):
#         pts, time_base = await self.next_timestamp()
#
#         frame = self.camera.read()
#         if frame is None:
#             await asyncio.sleep(0.03)
#             return await self.recv()
#
#         processed, violation = self.ppe_model.predict(frame)
#
#         rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
#         video_frame = VideoFrame.from_ndarray(rgb, format="rgb24")
#         video_frame.pts = pts
#         video_frame.time_base = time_base
#         return video_frame


import cv2
import asyncio
from av import VideoFrame
from aiortc import VideoStreamTrack


class PPEVideoTrack(VideoStreamTrack):
    # CRITICAL: This attribute tells the browser to treat this as video
    kind = "video"

    def __init__(self, camera, ppe_model):
        super().__init__()
        self.camera = camera
        self.ppe_model = ppe_model

    async def recv(self):
        # next_timestamp() handles the synchronization for the stream
        pts, time_base = await self.next_timestamp()

        # 1. Get frame from your CameraSource
        frame = self.camera.read()

        # 2. Handle empty frames (prevent the stream from crashing)
        if frame is None:
            # Wait for roughly 1 frame duration (30fps ~ 0.03s)
            await asyncio.sleep(0.03)
            # Recursively try to get the next frame
            return await self.recv()

        # 3. Run your PPE model (Inference)
        processed, violation = self.ppe_model.predict(frame)

        # 4. Prepare frame for WebRTC (aiortc/PyAV uses RGB)
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame