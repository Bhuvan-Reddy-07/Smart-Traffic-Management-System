import os
import sys
import cv2
import numpy as np
import subprocess
import pathlib
import time

COMMON_DIR = pathlib.Path(r"c:\Users\bhuva\Desktop\practice\ECS\ecs 1.0.1\common")
sys.path.insert(0, str(COMMON_DIR))
import utils as util

MODEL_PATH = pathlib.Path(r"c:\Users\bhuva\Desktop\practice\ECS\ecs 1.0.2\Smart_Traffic_Management_System\Smart_Traffic_Management_System\models\yolov5s.onnx")
DATA_DIR = pathlib.Path(r"c:\Users\bhuva\Desktop\practice\ECS\ecs 1.0.1\datas")
OUTPUT_DIR = pathlib.Path(r"c:\Users\bhuva\Desktop\practice\ECS\ecs 1.0.2\Smart_Traffic_Management_System\Smart_Traffic_Management_System\data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading YOLOv5s ONNX model from: {MODEL_PATH}")
net = cv2.dnn.readNet(str(MODEL_PATH))
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
out_layers = net.getUnconnectedOutLayersNames()

# Class colors (BGR)
COLORS = {
    "car": (80, 220, 100),       # Emerald green
    "truck": (50, 160, 255),     # Amber/Orange
    "bus": (255, 100, 180),      # Violet / Purple
    "motorbike": (240, 200, 40), # Cyan / Sky blue
    "bicycle": (200, 240, 80),   # Lime
    "person": (180, 180, 180),   # Gray
}

def draw_modern_box(frame, x1, y1, x2, y2, class_name, conf):
    color = COLORS.get(class_name, (0, 255, 255))
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    # Box outline
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    
    # Corner brackets for futuristic look
    corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
    if corner_len > 3:
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 4, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 4, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 4, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y2 - (y2 - y1) + corner_len), color, 4, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 4, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 4, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 4, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 4, cv2.LINE_AA)

    # Label pill
    label = f"{class_name.upper()} {int(conf * 100)}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    label_y = max(y1 - 6, th + 6)
    cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw + 8, label_y + 4), color, -1)
    cv2.putText(frame, label, (x1 + 4, label_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

def detect_frame(frame, target_w=960, target_h=540, lane_name="LANE"):
    frame_resized = cv2.resize(frame, (target_w, target_h))
    h, w = frame_resized.shape[:2]

    # YOLO inference
    blob = cv2.dnn.blobFromImage(frame_resized, 1 / 255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)
    dets = util.modify(outs, confThreshold=0.35, nmsThreshold=0.45, objThreshold=0.35)
    
    # Process detections
    ratioh, ratiow = h / 320, w / 320
    class_ids, confs, boxes = [], [], []
    for out in dets:
        for d in out:
            scores = d[5:]
            cid = int(np.argmax(scores))
            conf = float(scores[cid])
            if conf > 0.35 and float(d[4]) > 0.35:
                cx = int(d[0] * ratiow)
                cy = int(d[1] * ratioh)
                bw = int(d[2] * ratiow)
                bh = int(d[3] * ratioh)
                left = int(cx - bw / 2)
                top = int(cy - bh / 2)
                class_ids.append(cid)
                confs.append(conf)
                boxes.append([left, top, bw, bh])

    indices = cv2.dnn.NMSBoxes(boxes, confs, 0.35, 0.4)
    v_count = 0
    vehicle_types = {"car", "truck", "bus", "motorbike", "bicycle"}
    
    annotated = frame_resized.copy()
    if len(indices) > 0:
        for idx in indices:
            i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else int(idx)
            cid = class_ids[i]
            cname = util.COCO_CLASSES[cid] if cid < len(util.COCO_CLASSES) else "object"
            if cname in vehicle_types:
                v_count += 1
                b = boxes[i]
                draw_modern_box(annotated, b[0], b[1], b[0] + b[2], b[1] + b[3], cname, confs[i])

    # Overlay HUD badge
    hud_h = 30
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)
    cv2.putText(annotated, f"LIVE AI PERCEPTION | {lane_name}", (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 189, 248), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"VEHICLES DETECTED: {v_count}", (w - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (74, 222, 128), 1, cv2.LINE_AA)

    return annotated, v_count

def process_single_video(input_path, output_mp4, lane_name, max_frames=250):
    print(f"\nProcessing {lane_name} from {input_path} (max {max_frames} frames)...")
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error opening {input_path}")
        return

    fps = 20.0
    w, h = 960, 540
    temp_avi = output_mp4.parent / f"temp_{output_mp4.stem}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(str(temp_avi), fourcc, fps, (w, h))

    count = 0
    t0 = time.time()
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        annotated, v_count = detect_frame(frame, target_w=w, target_h=h, lane_name=lane_name)
        out.write(annotated)
        count += 1
        if count % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {lane_name}: {count}/{max_frames} frames ({count/elapsed:.1f} fps)")

    cap.release()
    out.release()

    print(f"  Encoding {output_mp4.name} with ffmpeg (h264)...")
    cmd = [
        "ffmpeg", "-y", "-i", str(temp_avi),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "23",
        "-movflags", "+faststart",
        str(output_mp4)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if temp_avi.exists():
        temp_avi.unlink()
    print(f"  Done: {output_mp4} ({output_mp4.stat().st_size / 1024 / 1024:.2f} MB)")

def create_quad_grid_video(output_quad_mp4, max_frames=250):
    print(f"\nGenerating 4-Way Quad Matrix video ({output_quad_mp4.name})...")
    caps = [
        cv2.VideoCapture(str(DATA_DIR / "video1.mp4")),
        cv2.VideoCapture(str(DATA_DIR / "video5.mp4")),
        cv2.VideoCapture(str(DATA_DIR / "video2.mp4")),
        cv2.VideoCapture(str(DATA_DIR / "video3.mp4")),
    ]
    lane_titles = ["NORTH (L1)", "EAST (L2)", "SOUTH (L3)", "WEST (L4)"]

    w, h = 480, 270
    total_w, total_h = w * 2, h * 2
    fps = 20.0
    temp_avi = output_quad_mp4.parent / f"temp_{output_quad_mp4.stem}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(str(temp_avi), fourcc, fps, (total_w, total_h))

    count = 0
    t0 = time.time()
    while count < max_frames:
        quad_frames = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            annotated, _ = detect_frame(frame, target_w=w, target_h=h, lane_name=lane_titles[i])
            quad_frames.append(annotated)
            
        top_row = np.hstack([quad_frames[0], quad_frames[1]])
        bot_row = np.hstack([quad_frames[2], quad_frames[3]])
        full_quad = np.vstack([top_row, bot_row])
        out.write(full_quad)

        count += 1
        if count % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Quad Matrix: {count}/{max_frames} frames ({count/elapsed:.1f} fps)")

    for cap in caps:
        cap.release()
    out.release()

    print(f"  Encoding {output_quad_mp4.name} with ffmpeg...")
    cmd = [
        "ffmpeg", "-y", "-i", str(temp_avi),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "23",
        "-movflags", "+faststart",
        str(output_quad_mp4)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if temp_avi.exists():
        temp_avi.unlink()
    print(f"  Done Quad Matrix: {output_quad_mp4} ({output_quad_mp4.stat().st_size / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    t_start = time.time()
    
    # Process 4 individual detected lane videos
    process_single_video(DATA_DIR / "video1.mp4", OUTPUT_DIR / "video1_detected.mp4", "NORTH APPROACH (L1)", max_frames=200)
    process_single_video(DATA_DIR / "video5.mp4", OUTPUT_DIR / "video5_detected.mp4", "EAST CORRIDOR (L2)", max_frames=200)
    process_single_video(DATA_DIR / "video2.mp4", OUTPUT_DIR / "video2_detected.mp4", "SOUTH JUNCTION (L3)", max_frames=200)
    process_single_video(DATA_DIR / "video3.mp4", OUTPUT_DIR / "video3_detected.mp4", "WEST EXPRESSWAY (L4)", max_frames=200)

    # Process 4-way unified quad matrix video
    create_quad_grid_video(OUTPUT_DIR / "quad_detected.mp4", max_frames=200)

    print(f"\nAll videos successfully generated in {time.time() - t_start:.1f}s!")
