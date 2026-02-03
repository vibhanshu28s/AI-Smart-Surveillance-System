import cv2
import os
import time


class CameraSource:
    def __init__(self, rtsp_url=None, webcam_index=0, width=640):
        self.rtsp_url = rtsp_url
        self.webcam_index = webcam_index
        self.cap = None
        self.target_width = width  # Downscale for faster motion processing

        # Force UDP for RTSP to reduce latency
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

        self._open_camera()

    def _open_camera(self):
        # 1. Try RTSP first
        if self.rtsp_url:
            print(f"[INFO] Connecting to RTSP: {self.rtsp_url}")
            self.cap = cv2.VideoCapture(self.rtsp_url)
            # Give the stream a moment to buffer
            time.sleep(2)

        # 2. Fallback to webcam
        if not self.cap or not self.cap.isOpened():
            print("[WARN] RTSP unavailable. Falling back to webcam.")
            self.cap = cv2.VideoCapture(self.webcam_index)

        # 3. Final check
        if not self.cap.isOpened():
            raise RuntimeError("❌ Error: Could not open any camera source.")

        print("[INFO] Camera stream initialized")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Resize for performance: processing motion on 640px is
        # much faster than 1080px and usually more accurate (less noise)
        h, w = frame.shape[:2]
        aspect_ratio = h / w
        target_height = int(self.target_width * aspect_ratio)

        frame = cv2.resize(frame, (self.target_width, target_height))
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
            print("[INFO] Camera released")