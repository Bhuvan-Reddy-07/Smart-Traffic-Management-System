# Data Collection Script
"""
Data Collection Engine
Captures video streams and individual frames from connected IP/USB cameras or pre-recorded
traffic video datasets, preprocessing and archiving frames into the data directory for AI processing.
"""
import os
import pathlib
import time
import cv2
import numpy as np
from typing import Optional, Generator, Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

def get_video_source(source_index_or_path: Any = 0) -> cv2.VideoCapture:
    """
    Attempts to open specified camera index or video file.
    Falls back to bundled sample video or synthetic generator if camera is unavailable.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample_video = DATA_DIR / "video_sample.mp4"

    # Try requested source
    cap = cv2.VideoCapture(source_index_or_path)
    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return cap
        cap.release()

    # Fallback to local sample video
    if sample_video.exists():
        cap = cv2.VideoCapture(str(sample_video))
        if cap.isOpened():
            return cap

    return None

def generate_synthetic_frame(frame_num: int = 0) -> np.ndarray:
    """
    Generates a realistic simulated multi-lane intersection camera frame with moving vehicles.
    """
    frame = np.full((720, 1280, 3), 35, dtype=np.uint8)
    
    # Asphalt road (horizontal and vertical cross)
    cv2.rectangle(frame, (440, 0), (840, 720), (55, 55, 55), -1)
    cv2.rectangle(frame, (0, 220), (1280, 500), (55, 55, 55), -1)

    # Road boundaries & markings
    cv2.line(frame, (640, 0), (640, 220), (255, 255, 255), 2)
    cv2.line(frame, (640, 500), (640, 720), (255, 255, 255), 2)
    cv2.line(frame, (0, 360), (440, 360), (255, 255, 255), 2)
    cv2.line(frame, (840, 360), (1280, 360), (255, 255, 255), 2)

    # Intersection box
    cv2.rectangle(frame, (440, 220), (840, 500), (45, 45, 45), -1)
    
    # Moving cars simulation
    speed = 7
    # Car 1 - moving south
    y1 = int((frame_num * speed) % 700)
    cv2.rectangle(frame, (520, y1), (580, y1 + 90), (0, 140, 255), -1)
    cv2.putText(frame, "CAR", (530, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Car 2 - moving east
    x2 = int((frame_num * (speed + 2)) % 1200)
    cv2.rectangle(frame, (x2, 260), (x2 + 100, 320), (50, 205, 50), -1)
    cv2.putText(frame, "CAR", (x2 + 30, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Ambulance (Emergency vehicle)
    if (frame_num // 60) % 2 == 1:
        cv2.rectangle(frame, (700, 520), (770, 640), (0, 0, 255), -1)
        cv2.putText(frame, "AMBULANCE", (705, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame

def collect_data(max_frames: Optional[int] = 5, delay: float = 0.5) -> int:
    """
    Collects and archives traffic camera frames into the data/ directory.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cap = get_video_source(0)
    captured_count = 0

    print(f"[Collector] Archiving frames into: {DATA_DIR}")
    
    frame_counter = 0
    try:
        while True:
            if max_frames is not None and captured_count >= max_frames:
                break

            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    # Loop video if reached end
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
            else:
                frame = generate_synthetic_frame(frame_counter)
                ret = True

            if not ret or frame is None:
                break

            frame_path = DATA_DIR / f"frame_{int(time.time())}_{captured_count}.jpg"
            cv2.imwrite(str(frame_path), frame)
            captured_count += 1
            frame_counter += 1
            print(f"-> Saved frame #{captured_count}: {frame_path.name}")
            
            time.sleep(delay)
    finally:
        if cap is not None:
            cap.release()
            cv2.destroyAllWindows()

    print(f"[Collector] Completed. Total {captured_count} frames collected.")
    return captured_count


if __name__ == '__main__':
    print("=" * 60)
    print("SMART TRAFFIC SYSTEM - DATA COLLECTION MODULE")
    print("=" * 60)
    collect_data(max_frames=3, delay=0.2)