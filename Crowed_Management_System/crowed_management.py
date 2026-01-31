import cv2
from ultralytics import YOLO
import numpy as np
import time
import csv
import pygame

# --- INITIALIZE AUDIO ---
pygame.mixer.init()
try:
    alert_sound = pygame.mixer.Sound("audio.mp3")
except:
    print("Warning: audio.mp3 not found. Audio alerts will be skipped.")
    alert_sound = None

# --- MOUSE CALLBACK FUNCTION ---
roi_points = []


def draw_roi(event, x, y, flags, param):
    global roi_points
    if event == cv2.EVENT_LBUTTONDOWN and len(roi_points) < 4:
        roi_points.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:
        roi_points = []


# --- INITIALIZATION ---
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
cv2.namedWindow("Crowd Control")
cv2.setMouseCallback("Crowd Control", draw_roi)

CROWD_LIMIT = 1
WARNING_DURATION = 2  # Seconds of overcrowding before alarm plays
overcrowded_start_time = None
last_log_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    current_time = time.time()

    if len(roi_points) < 4:
        # Setup instructions
        for pt in roi_points: cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, "Click 4 points to set Zone", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        pts_array = np.array(roi_points, np.int32)
        cv2.polylines(frame, [pts_array], True, (255, 255, 0), 2)

        results = model(frame, conf=0.4, verbose=False)
        person_count = 0

        for box in results[0].boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = box.xyxy[0]
                cx, cy = int((x1 + x2) / 2), int(y2)

                if cv2.pointPolygonTest(pts_array, (cx, cy), False) >= 0:
                    person_count += 1
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # --- PERSISTENT ALERT LOGIC ---
        if person_count >= CROWD_LIMIT:
            if overcrowded_start_time is None:
                overcrowded_start_time = current_time  # Start the clock

            elapsed_overcrowded = current_time - overcrowded_start_time

            # Change UI to red and show warning timer
            cv2.putText(frame, f"WARNING: {int(elapsed_overcrowded)}s", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            if elapsed_overcrowded >= WARNING_DURATION:
                cv2.putText(frame, "!!! CLEAR THE AREA !!!", (200, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                if alert_sound and not pygame.mixer.get_busy():
                    alert_sound.play()
        else:
            overcrowded_start_time = None  # Reset clock if crowd disperses

        # Display Count
        cv2.putText(frame, f"Zone Count: {person_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Crowd Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()