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
        # We detect violations directly based on your "no_" labels
        violations = []
        if "no_helmet" in found_labels: violations.append("HELMET")
        if "no_vest" in found_labels: violations.append("VEST")
        if "no_gloves" in found_labels: violations.append("GLOVES")
        if "no_boots" in found_labels: violations.append("BOOTS")
        if "no_goggle" in found_labels: violations.append("GOGGLES")
        if "none" in found_labels: violations.append("ALL PPE")

        # Trigger if any violation labels were found
        violation_detected = len(violations) > 0

        # --- VISUAL ALERT ---
        if violation_detected:
            # Draw Red Header Bar
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

            # Create dynamic alert text based on detected "no_" labels
            alert_text = f"VIOLATION: MISSING {', '.join(violations)}"

            # Scale text if it's too long
            font_scale = 0.8 if len(alert_text) > 30 else 1.0

            cv2.putText(
                annotated_frame,
                alert_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # --- AUDIO & SAVE LOGIC ---
        if violation_detected:
            now = time.time()
            if now - self.last_alert_time > self.COOLDOWN_SECONDS:
                self.alert_sound.play()
                ts = time.strftime("%Y%m%d-%H%M%S")
                path = os.path.join(self.alert_dir, f"violation_{ts}.jpg")
                cv2.imwrite(path, annotated_frame)
                print(f"[ALERT] PPE violation → {path}")
                self.last_alert_time = now

        return annotated_frame, violation_detected