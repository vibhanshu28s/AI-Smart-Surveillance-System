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
        results = self.model(frame, stream=True)
        annotated_frame = frame.copy()

        found_labels = []
        for r in results:
            annotated_frame = r.plot()
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > self.CONFIRMATION_THRESHOLD:
                    label = self.model.names[int(box.cls[0])]
                    found_labels.append(label)

        # --- SMART LOGIC ---
        # 1. Check if a person is present
        person_present = "Person" in found_labels

        # 2. Define what "Safe" looks like (Modify these names based on your .onnx labels)
        has_helmet = "no_helmet" in found_labels
        has_vest = "no_vest" in found_labels
        has_glove = "no_gloves" in found_labels
        has_boots = "no_boots" in found_labels
        has_goggles = "no_goggles" in found_labels
        has_none = "none" in found_labels


        # 3. Logic: Alert if a person is present but missing gear
        # You can change 'and' to 'or' depending on if BOTH are required.
        violation_detected = person_present and (not has_helmet or not has_vest)

        # --- VISUAL ALERT ---
        if violation_detected:
            # Draw Red Header
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

            # Identify what is missing for the text
            missing = []
            if not has_helmet: missing.append("HELMET")
            if not has_vest: missing.append("VEST")
            alert_text = f"MISSING: {', '.join(missing)}" if person_present else "PPE VIOLATION"

            cv2.putText(annotated_frame, alert_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        # --- AUDIO & SAVE LOGIC ---
        if violation_detected:
            now = time.time()
            if now - self.last_alert_time > self.COOLDOWN_SECONDS:
                self.alert_sound.play()
                ts = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(f"{self.alert_dir}/violation_{ts}.jpg", annotated_frame)
                self.last_alert_time = now

        return annotated_frame, violation_detected
