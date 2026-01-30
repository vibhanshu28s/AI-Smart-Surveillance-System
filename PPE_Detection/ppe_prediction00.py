import cv2
import time
import os
from ultralytics import YOLO
import pygame

class PPEPredictor:
    def __init__(self):
        # Audio
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound("alert.mp3")

        # Model
        self.model = YOLO("best.onnx", task="detect")

        # Settings
        self.CONFIRMATION_THRESHOLD = 0.5
        self.COOLDOWN_SECONDS = 5
        self.last_alert_time = 0

        # Screenshot dir
        self.alert_dir = "alerts_screenshots"
        os.makedirs(self.alert_dir, exist_ok=True)

    def predict(self, frame):
        """
        Input  : BGR frame
        Output : (annotated_frame, violation_detected)
        """
        results = self.model(frame, stream=True)
        annotated_frame = frame.copy()
        violation_detected = False

        for r in results:
            annotated_frame = r.plot()
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls]

                if conf > self.CONFIRMATION_THRESHOLD and ("NO" in label or label == "Person"):
                    violation_detected = True

        # Handle alert logic (stateful but isolated)
        if violation_detected:
            now = time.time()
            if now - self.last_alert_time > self.COOLDOWN_SECONDS:
                self.alert_sound.play()

                ts = time.strftime("%Y%m%d-%H%M%S")
                path = f"{self.alert_dir}/violation_{ts}.jpg"
                cv2.imwrite(path, annotated_frame)

                print(f"[ALERT] PPE violation → {path}")
                self.last_alert_time = now

        return annotated_frame, violation_detected
