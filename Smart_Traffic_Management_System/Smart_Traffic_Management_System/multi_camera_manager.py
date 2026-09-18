"""
Multi-Camera Traffic Perception Engine
Integrates the 4-lane YOLOv5s ONNX perception pipeline from ecs 1.0.1 with the
dynamic smart traffic controller and web dashboard from ecs 1.0.2.
Processes 4 video streams simultaneously, stitches a 2x2 multi-camera grid,
and streams real-time telemetry to the system.
"""
import os
import sys
import time
import pathlib
import threading
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR / "common"
DATA_DIR = SCRIPT_DIR / "data"
MODELS_DIR = SCRIPT_DIR / "models"

# Ensure common is on sys.path
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import utils as util

# Lane mapping
LANE_CONFIG = {
    1: {"name": "North", "video": "video1.mp4", "pos": "Top-Left"},
    2: {"name": "East",  "video": "video5.mp4", "pos": "Top-Right"},
    3: {"name": "South", "video": "video2.mp4", "pos": "Bottom-Left"},
    4: {"name": "West",  "video": "video3.mp4", "pos": "Bottom-Right"}
}

class MultiCameraManager:
    def __init__(self):
        self.caps: Dict[int, cv2.VideoCapture] = {}
        self.net = None
        self.out_layers = []
        self.lock = threading.Lock()
        
        # Current detection states
        self.lane_counts = {"North": 0, "East": 0, "South": 0, "West": 0}
        self.latest_frames = {}
        self.latest_grid_frame = None
        self.running = False
        self.worker_thread = None

        self._init_model()
        self._init_captures()

    def _init_model(self):
        """Loads YOLOv5s ONNX model."""
        model_paths = [
            MODELS_DIR / "yolov5s.onnx",
            SCRIPT_DIR.parent.parent / "ecs 1.0.1" / "models" / "yolov5s.onnx"
        ]
        for p in model_paths:
            if p.exists():
                try:
                    self.net = cv2.dnn.readNet(str(p))
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    self.out_layers = self.net.getUnconnectedOutLayersNames()
                    print(f"[MultiCam] Loaded YOLOv5s model: {p.name}")
                    break
                except Exception as e:
                    print(f"[MultiCam] Warning loading model: {e}")

    def _init_captures(self):
        """Opens video captures for all 4 lanes."""
        for lane_id, cfg in LANE_CONFIG.items():
            video_file = DATA_DIR / cfg["video"]
            if video_file.exists():
                cap = cv2.VideoCapture(str(video_file))
                if cap.isOpened():
                    self.caps[lane_id] = cap
                    print(f"[MultiCam] Lane {lane_id} ({cfg['name']}) -> {cfg['video']}")
                else:
                    print(f"[MultiCam] Warning: Could not open {video_file}")
            else:
                print(f"[MultiCam] Notice: {video_file} not found, will use synthetic fallback")

    def _read_lane_frame(self, lane_id: int) -> np.ndarray:
        cap = self.caps.get(lane_id)
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if ret and frame is not None:
                return frame

        # Fallback frame
        h, w = 720, 1280
        fallback = np.full((h, w, 3), 40, dtype=np.uint8)
        cv2.putText(fallback, f"LANE {lane_id} CAMERA FEED", (w//4, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 2)
        return fallback

    def start(self):
        """Starts background frame processing thread."""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._process_loop, daemon=True)
            self.worker_thread.start()
            print("[MultiCam] Multi-camera processing engine started.")

    def _process_loop(self):
        lanes_obj = util.Lanes([
            util.Lane(0, None, 1),
            util.Lane(0, None, 2),
            util.Lane(0, None, 3),
            util.Lane(0, None, 4),
        ])

        inference_interval = 3  # Run YOLO every 3 frames for smooth 25 FPS video playback
        frame_counter = 0

        while self.running:
            # 1. Grab frames from all 4 cameras
            raw_frames = {}
            for lane_id in [1, 2, 3, 4]:
                raw_frames[lane_id] = self._read_lane_frame(lane_id)

            # Assign frames to lanes_obj
            for lane in lanes_obj.getLanes():
                lane.frame = raw_frames[lane.lane_number]

            # 2. Periodic YOLO inference on all 4 lanes
            if frame_counter % inference_interval == 0 and self.net is not None:
                try:
                    lanes_obj = util.final_output(self.net, self.out_layers, lanes_obj)
                except Exception as e:
                    # Non-fatal inference catch
                    pass

            frame_counter += 1

            # 3. Store processed frames and counts
            with self.lock:
                for lane in lanes_obj.getLanes():
                    lane_name = LANE_CONFIG[lane.lane_number]["name"]
                    self.lane_counts[lane_name] = lane.count
                    self.latest_frames[lane.lane_number] = lane.frame.copy() if lane.frame is not None else None

            time.sleep(0.03)

    def get_latest_counts(self) -> Dict[str, int]:
        with self.lock:
            return dict(self.lane_counts)

    def generate_composite_grid(self, sim_state: Dict[str, Any]) -> np.ndarray:
        """
        Stitches the 4 video feeds into a 2x2 grid with dynamic traffic light borders,
        countdown clocks, and detection stats.
        """
        with self.lock:
            f1 = self.latest_frames.get(1)
            f2 = self.latest_frames.get(2)
            f3 = self.latest_frames.get(3)
            f4 = self.latest_frames.get(4)

        if f1 is None or f2 is None or f3 is None or f4 is None:
            # Return placeholder if still initializing
            placeholder = np.full((720, 1280, 3), 20, dtype=np.uint8)
            cv2.putText(placeholder, "INITIALIZING 4-WAY CAMERA ARRAY...", (320, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            return placeholder

        # Target dimensions for each quadrant: 640x360 -> Full grid: 1280x720
        quad_w, quad_h = 640, 360
        processed_quads = {}

        for lane_id in [1, 2, 3, 4]:
            lane_name = LANE_CONFIG[lane_id]["name"]
            raw_q = self.latest_frames.get(lane_id)
            q_frame = cv2.resize(raw_q, (quad_w, quad_h))
            
            # Look up state for this lane from sim_state
            lane_info = sim_state.get("lanes", {}).get(lane_name, {})
            current_state = lane_info.get("state", "RED")
            count = lane_info.get("count", self.lane_counts.get(lane_name, 0))
            is_emergency = lane_info.get("emergency", False)
            timer = sim_state.get("timer_remaining", 0)

            # Border and indicator color based on traffic signal
            if is_emergency:
                theme_color = (0, 0, 255)  # Bright Red
                status_text = "EMERGENCY PREEMPTION ACTIVE"
            elif current_state == "GREEN":
                theme_color = (0, 220, 100)  # Bright Emerald
                status_text = f"GREEN - ACTIVE ({timer}s)"
            elif current_state == "YELLOW":
                theme_color = (0, 200, 255)  # Amber
                status_text = f"YELLOW - CLEARING ({timer}s)"
            else:
                theme_color = (0, 50, 220)   # Red
                status_text = "RED - QUEUED"

            # Draw glowing signal border around quadrant
            cv2.rectangle(q_frame, (0, 0), (quad_w - 1, quad_h - 1), theme_color, 3)

            # Top HUD banner
            cv2.rectangle(q_frame, (0, 0), (quad_w, 36), (15, 20, 30), -1)
            cv2.putText(q_frame, f"LANE {lane_id}: {lane_name.upper()} | {status_text}", 
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, theme_color, 2, cv2.LINE_AA)

            # Bottom vehicle counter badge
            badge_text = f"VEHICLES DETECTED: {count}"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(q_frame, (10, quad_h - 35), (20 + tw, quad_h - 10), (10, 15, 25), -1)
            cv2.putText(q_frame, badge_text, (15, quad_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            processed_quads[lane_id] = q_frame

        # Tile 2x2 grid
        top_row = np.concatenate((processed_quads[1], processed_quads[2]), axis=1)
        bottom_row = np.concatenate((processed_quads[3], processed_quads[4]), axis=1)
        full_grid = np.concatenate((top_row, bottom_row), axis=0)

        # Center crossroads watermark badge
        center_w, center_h = 240, 44
        cx = (1280 - center_w) // 2
        cy = (720 - center_h) // 2
        cv2.rectangle(full_grid, (cx, cy), (cx + center_w, cy + center_h), (10, 15, 25), -1)
        cv2.rectangle(full_grid, (cx, cy), (cx + center_w, cy + center_h), (0, 220, 255), 1)
        cv2.putText(full_grid, "AI TRAFFIC MATRIX (4-WAY)", (cx + 15, cy + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 255), 2, cv2.LINE_AA)

        return full_grid

    def get_single_lane_frame(self, lane_id: int, sim_state: Dict[str, Any]) -> np.ndarray:
        """Returns single high-resolution stream for selected lane."""
        with self.lock:
            frame = self.latest_frames.get(lane_id)
        if frame is None:
            return np.full((720, 1280, 3), 30, dtype=np.uint8)

        out = cv2.resize(frame, (1280, 720))
        lane_name = LANE_CONFIG[lane_id]["name"]
        lane_info = sim_state.get("lanes", {}).get(lane_name, {})
        state = lane_info.get("state", "RED")
        count = lane_info.get("count", self.lane_counts.get(lane_name, 0))

        color = (0, 220, 100) if state == "GREEN" else ((0, 200, 255) if state == "YELLOW" else (0, 50, 220))
        cv2.rectangle(out, (0, 0), (1280, 48), (15, 20, 30), -1)
        cv2.putText(out, f"LANE {lane_id}: {lane_name.upper()} CAMERA | SIGNAL: {state} | CARS: {count}", 
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        return out


# Global singleton instance
multi_camera_manager = MultiCameraManager()
multi_camera_manager.start()
