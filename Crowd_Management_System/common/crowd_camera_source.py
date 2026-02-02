import cv2
import os
import time

class CameraSource:
    def __init__(self, rtsp_url=None, webcam_index=0):
        self.rtsp_url = rtsp_url
        self.webcam_index = webcam_index
        self.cap = None

        # Force UDP for RTSP (important)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

        self._open_camera()

    def _open_camera(self):
        # 1️⃣ Try RTSP first
        if self.rtsp_url:
            print("[INFO] Trying RTSP stream...")
            self.cap = cv2.VideoCapture(self.rtsp_url)
            time.sleep(1)

        # 2️⃣ Fallback to webcam
        if not self.cap or not self.cap.isOpened():
            print("[WARN] RTSP failed. Falling back to webcam.")
            self.cap = cv2.VideoCapture(self.webcam_index)

        # 3️⃣ Final check
        if not self.cap.isOpened():
            raise RuntimeError("❌ No camera source available")

        print("[INFO] Camera source opened successfully")

    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap:
            self.cap.release()
