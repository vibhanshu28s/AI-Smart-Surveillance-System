import cv2
import time
import os
from ultralytics import YOLO
import pygame


class CrowdManager:
    def __init__(self):
        # 1. Improved Audio Init - Specific frequency prevents silent failures
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()

        # Ensure the path to the mp3 is correct
        try:
            self.alert_sound = pygame.mixer.Sound("alert.mp3")
        except Exception as e:
            print(f"[WARN] Could not load alert.mp3: {e}")
            self.alert_sound = None

        self.model = YOLO("yolov8n.pt")
        self.CONFIRMATION_THRESHOLD = 0.4
        self.CROWD_THRESHOLD = 1
        self.COOLDOWN_SECONDS = 5  # Reduced for easier testing

        # 2. Correct Init: ensures first detection triggers immediately
        self.last_alert_time = 0

        # 3. Use Absolute Paths for Screenshot Directory
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.alert_dir = os.path.join(self.base_dir, "crowd_alerts")
        os.makedirs(self.alert_dir, exist_ok=True)

        self.roi = None

    def update_roi(self, data):
        self.roi = (int(data['x1']), int(data['y1']), int(data['x2']), int(data['y2']))

    def predict(self, frame):
        results = self.model(frame, stream=True)
        annotated_frame = frame.copy()
        person_count = 0

        if self.roi:
            x1, y1, x2, y2 = self.roi
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > self.CONFIRMATION_THRESHOLD:
                    label = self.model.names[int(box.cls[0])]
                    if label == "person":
                        coords = box.xyxy[0].tolist()
                        bx1, by1, bx2, by2 = map(int, coords)
                        cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2

                        if self.roi:
                            rx1, ry1, rx2, ry2 = self.roi
                            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                                person_count += 1
                                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                        else:
                            person_count += 1

        # Logic: Violation if count is GREATER than threshold
        violation_detected = person_count > self.CROWD_THRESHOLD

        # Visuals
        status_color = (0, 0, 255) if violation_detected else (0, 255, 0)
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), status_color, -1)
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
        cv2.putText(annotated_frame, f"ZONE COUNT: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 4. FIXED ACTION LOGIC
        if violation_detected:
            now = time.time()
            if now - self.last_alert_time > self.COOLDOWN_SECONDS:
                # Trigger Sound
                if self.alert_sound:
                    self.alert_sound.play()

                # Trigger Save with Absolute Path
                ts = time.strftime("%Y%m%d-%H%M%S")
                save_path = os.path.join(self.alert_dir, f"crowd_violation_{ts}.jpg")

                success = cv2.imwrite(save_path, annotated_frame)
                if success:
                    print(f"[ALERT] Screenshot saved to: {save_path}")
                else:
                    print(f"[ERROR] Could not save screenshot to {save_path}")

                self.last_alert_time = now

        return annotated_frame, violation_detected