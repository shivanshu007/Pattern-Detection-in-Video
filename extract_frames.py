import cv2
import os

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        cv2.imwrite(f"{output_folder}/frame_{count}.jpg", frame)
        count += 1
    cap.release()
