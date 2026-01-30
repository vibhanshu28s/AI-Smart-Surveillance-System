import cv2
import os

def webcam():
    ip_camera_url = "rtsp://username:password@192.168.1.64/stream"
    webcam_index = 0

    # Optional: Speed up RTSP connection
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

    print("Attempting to open IP Camera...")
    cap = cv2.VideoCapture(ip_camera_url)

    if not cap.isOpened():
        print("IP Camera offline. Switching to Webcam...")
        cap = cv2.VideoCapture(webcam_index)

        if not cap.isOpened():
            print("Error: No camera source found.")
            return None # Return None instead of exiting to handle it in main

    # We remove the while loop from here because ppe_prediction will handle the loop
    return cap

