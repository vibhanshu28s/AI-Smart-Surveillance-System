def ppe_prediction():
    from PPE_Detection.webRTC import webcam as wc
    import cv2
    from ultralytics import YOLO
    import pygame
    import time
    import os

    # Initialize Audio
    pygame.mixer.init()
    alert_sound = pygame.mixer.Sound("PPE_Detection-11/alert.mp3")  # Ensure you have an 'alert.wav' file

    # Load Model
    model = YOLO("PPE_Detection-11/best.onnx", task='detect')

    # Settings
    CONFIRMATION_THRESHOLD = 0.5
    COOLDOWN_SECONDS = 5  # Time to wait before taking another screenshot/alert
    last_alert_time = 0

    # Create a folder for screenshots
    if not os.path.exists('PPE_Detection-11/alerts_screenshots'):
        os.makedirs('PPE_Detection-11/alerts_screenshots')

    cap = wc()

    if cap is not None and cap.isOpened():
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            results = model(frame, stream=True)
            annotated_frame = frame.copy()
            violation_detected = False

            for r in results:
                annotated_frame = r.plot()
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls]

                    # Trigger logic: If 'No-Helmet' or 'No-Vest' is detected
                    if conf > CONFIRMATION_THRESHOLD and ("NO" in label or label == "Person"):
                        violation_detected = True

            # Handle Alert and Screenshot
            if violation_detected:
                current_time = time.time()
                if current_time - last_alert_time > COOLDOWN_SECONDS:
                    # 1. Play Sound
                    alert_sound.play()

                    # 2. Save Screenshot
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    file_path = f"PPE_Detection-11/alerts_screenshots/violation_{timestamp}.jpg"
                    cv2.imwrite(file_path, annotated_frame)
                    print(f"Violation Detected! Screenshot saved: {file_path}")

                    last_alert_time = current_time

            cv2.imshow("PPE Smart Surveillance", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ppe_prediction()