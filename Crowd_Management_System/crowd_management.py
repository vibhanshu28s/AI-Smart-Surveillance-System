import cv2
import time
import os
from ultralytics import YOLO
import pygame


class CrowdManager:
    def __init__(self):
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound("alert.mp3")
        self.model = YOLO("yolov8n.pt")
        self.CONFIRMATION_THRESHOLD = 0.4
        self.CROWD_THRESHOLD = 1
        self.COOLDOWN_SECONDS = 10
        self.last_alert_time = 0

        # Initial ROI is None (counts full screen)
        self.roi = None
        self.alert_dir = "crowd_alerts"
        os.makedirs(self.alert_dir, exist_ok=True)

    def update_roi(self, data):
        """Updates ROI from client data: {'x1':.., 'y1':.., 'x2':.., 'y2':..}"""
        self.roi = (int(data['x1']), int(data['y1']), int(data['x2']), int(data['y2']))

    def predict(self, frame):
        results = self.model(frame, stream=True)
        annotated_frame = frame.copy()
        person_count = 0

        # Draw the ROI rectangle if set by the client
        if self.roi:
            x1, y1, x2, y2 = self.roi
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(annotated_frame, "SELECTED ZONE", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                label = self.model.names[int(box.cls[0])]

                if label == "person" and conf > self.CONFIRMATION_THRESHOLD:
                    coords = box.xyxy[0].tolist()
                    bx1, by1, bx2, by2 = map(int, coords)

                    # Logic: Check if center of person is inside Client's ROI
                    cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2

                    if self.roi:
                        rx1, ry1, rx2, ry2 = self.roi
                        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                            person_count += 1
                            cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    else:
                        person_count += 1  # Default count if no ROI drawn

        # Visual/Alert logic remains the same
        violation_detected = person_count > self.CROWD_THRESHOLD
        status_color = (0, 0, 255) if violation_detected else (0, 255, 0)

        # Draw status bar and count
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), status_color, -1)
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
        cv2.putText(annotated_frame, f"ZONE COUNT: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return annotated_frame, violation_detected