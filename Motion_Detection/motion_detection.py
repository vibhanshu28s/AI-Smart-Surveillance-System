import cv2
import time
import os
import pygame
import numpy as np
from ultralytics import YOLO


class MotionPredictor:
    def __init__(self):
        pygame.mixer.init()
        try:
            self.alert_sound = pygame.mixer.Sound("alert.mp3")
        except:
            self.alert_sound = None

        # Initialize YOLOv8
        self.model = YOLO("yolov8n.pt")

        # Motion Settings (Working absdiff logic)
        self.prev_gray = None
        self.THRESHOLD = 40
        self.MIN_MOTION_COUNT = 5500

        # Recording Settings (Working MJPG logic)
        self.COOLDOWN_SECONDS = 3
        self.fps = 20.0
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        # Target folder as requested
        self.output_dir = "motion_alert"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.video_writer = None
        self.last_detection_time = 0
        self.is_recording = False

    def predict(self, frame):
        current_time = time.time()
        annotated_frame = frame.copy()

        # A. Detect Motion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return annotated_frame, False

        diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(diff, self.THRESHOLD, 255, cv2.THRESH_BINARY)
        motion_count = cv2.countNonZero(thresh)
        self.prev_gray = gray

        motion_detected = motion_count > self.MIN_MOTION_COUNT

        # B. Conditional YOLO Detection (Detect Person + Motion)
        if motion_detected or self.is_recording:
            results = self.model(frame, conf=0.5, verbose=False)

            # Person class index is 0 in YOLOv8
            person_found = any(int(box.cls[0]) == 0 for r in results for box in r.boxes)

            # Start Recording if both Person is detected AND Motion is present
            if person_found and motion_detected:
                self.last_detection_time = current_time
                annotated_frame = results[0].plot()

                if not self.is_recording:
                    self._start_recording(frame)

            # C. Active Recording Handler
            if self.is_recording:
                if current_time - self.last_detection_time < self.COOLDOWN_SECONDS:
                    cv2.putText(annotated_frame, "REC: MOVING PERSON", (20, 50), 1, 2, (0, 0, 255), 2)
                    self.video_writer.write(annotated_frame)
                else:
                    self._stop_recording()

        return annotated_frame, self.is_recording

    def _start_recording(self, frame):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"person_motion_{timestamp}.avi"
        full_path = os.path.join(self.output_dir, filename)

        height, width = frame.shape[:2]
        self.video_writer = cv2.VideoWriter(full_path, self.fourcc, self.fps, (width, height))

        if self.video_writer.isOpened():
            self.is_recording = True
            print(f"[ALERT] Starting Record: {full_path}")
            if self.alert_sound:
                self.alert_sound.play()
        else:
            self.is_recording = False

    def _stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            print(f"File saved in {self.output_dir}")
        self.is_recording = False