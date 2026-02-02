import cv2
import time
import os
from ultralytics import YOLO
import pygame


class CrowdManager:
    def __init__(self):
        # Audio Initialization
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound("alert.mp3")

        # Model - Assuming a standard YOLO model (like yolov8n.pt or your custom model)
        # If using your custom .onnx, ensure 'person' is a detectable class.
        self.model = YOLO("yolov8n.pt")

        # Settings
        self.CONFIRMATION_THRESHOLD = 0.4
        self.CROWD_THRESHOLD = 1  # Alert if more than 15 people are detected
        self.COOLDOWN_SECONDS = 10
        self.last_alert_time = 0

        # Screenshot dir
        self.alert_dir = "crowd_alerts"
        os.makedirs(self.alert_dir, exist_ok=True)

    def predict(self, frame):
        results = self.model(frame, stream=True)
        annotated_frame = frame.copy()

        person_count = 0

        for r in results:
            annotated_frame = r.plot()
            for box in r.boxes:
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                label = self.model.names[class_id]

                # Count only "person" detections above confidence threshold
                if label == "person" and conf > self.CONFIRMATION_THRESHOLD:
                    person_count += 1

        # --- CROWD LOGIC ---
        violation_detected = person_count > self.CROWD_THRESHOLD

        # Determine UI Colors (Green for OK, Red for Overcrowded)
        status_color = (0, 0, 255) if violation_detected else (0, 255, 0)
        status_text = "OVERCROWDED" if violation_detected else "NORMAL"

        # --- VISUAL FEEDBACK ---
        # Draw Header Bar
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 70), status_color, -1)
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

        # Draw Count and Status
        display_text = f"COUNT: {person_count} | STATUS: {status_text}"
        cv2.putText(
            annotated_frame,
            display_text,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

        # --- AUDIO & SAVE LOGIC ---
        if violation_detected:
            now = time.time()
            if now - self.last_alert_time > self.COOLDOWN_SECONDS:
                try:
                    self.alert_sound.play()
                except:
                    pass  # Prevent crash if audio file is missing

                ts = time.strftime("%Y%m%d-%H%M%S")
                path = os.path.join(self.alert_dir, f"crowd_limit_{ts}.jpg")
                cv2.imwrite(path, annotated_frame)
                print(f"[ALERT] Capacity exceeded: {person_count} people → {path}")
                self.last_alert_time = now

        return annotated_frame, violation_detected