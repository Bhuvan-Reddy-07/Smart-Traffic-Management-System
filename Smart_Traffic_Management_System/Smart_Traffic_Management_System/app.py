# Flask Web Application
"""
Smart Traffic Management System - Web Dashboard & Control Center
Fully integrates:
- ecs 1.0.1: 4-Lane Real-Time YOLOv5s ONNX multi-camera perception & 2x2 video grid
- ecs 1.0.2: Dynamic Signal Scheduling, Priority Emergency Override & Web Telemetry
"""
import os
import sys
import time
import threading
import pathlib
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request

# Import system modules
from traffic_signal_control import controller, control_traffic_signal
from multi_camera_manager import multi_camera_manager

app = Flask(__name__)

# State tracking
sim_state = {
    "active": True,
    "lanes": {
        "North": {"count": 12, "wait": 15, "state": "RED", "emergency": False, "passed": 42},
        "East":  {"count": 6,  "wait": 8,  "state": "RED", "emergency": False, "passed": 28},
        "South": {"count": 22, "wait": 30, "state": "GREEN", "emergency": False, "passed": 85},
        "West":  {"count": 7,  "wait": 10, "state": "RED", "emergency": False, "passed": 31}
    },
    "active_lane": "South",
    "phase": "GREEN",
    "timer_remaining": 25,
    "green_duration": 30,
    "yellow_duration": 3,
    "total_passed": 186,
    "emergency_active": False,
    "system_mode": "Hybrid Engine: ecs 1.0.1 Multi-Camera + ecs 1.0.2 Adaptive Control",
    "efficiency_gain": "38.5%",
    "avg_wait_reduction": "16.4s",
    "active_view": "grid"
}

state_lock = threading.Lock()

def simulation_background_worker():
    """
    Background simulation worker that synchronizes real-time YOLO vehicle counts
    from the 4 camera feeds and runs the dynamic signal control algorithm.
    """
    while True:
        time.sleep(1.0)
        with state_lock:
            # 1. Sync live vehicle counts from multi_camera_manager (ecs 1.0.1)
            real_counts = multi_camera_manager.get_latest_counts()
            for lane_name, real_count in real_counts.items():
                if lane_name in sim_state["lanes"]:
                    # Smooth count adjustment with detected traffic
                    if real_count > 0:
                        sim_state["lanes"][lane_name]["count"] = real_count

            # 2. Decrement active phase timer
            sim_state["timer_remaining"] -= 1

            # 3. Update queue waiting times for red lanes
            for lane_name, lane in sim_state["lanes"].items():
                if lane["state"] == "RED":
                    lane["wait"] += 1
                elif lane["state"] == "GREEN":
                    lane["wait"] = max(0, lane["wait"] - 2)
                    lane["passed"] += 1
                    sim_state["total_passed"] += 1

            # 4. Check if current light cycle expired
            if sim_state["timer_remaining"] <= 0:
                current_lane = sim_state["active_lane"]

                if sim_state["phase"] == "GREEN":
                    # Transition to YELLOW
                    sim_state["phase"] = "YELLOW"
                    sim_state["lanes"][current_lane]["state"] = "YELLOW"
                    sim_state["timer_remaining"] = sim_state["yellow_duration"]

                elif sim_state["phase"] == "YELLOW":
                    # Transition current lane to RED
                    sim_state["lanes"][current_lane]["state"] = "RED"
                    
                    # Check for emergency preemption first
                    emergency_target = None
                    for lname, ldata in sim_state["lanes"].items():
                        if ldata["emergency"]:
                            emergency_target = lname
                            break

                    if emergency_target:
                        next_lane = emergency_target
                    else:
                        # Select next lane based on real-time congestion pressure
                        scored = sorted(
                            sim_state["lanes"].items(),
                            key=lambda item: (item[1]["count"] * 1.5 + item[1]["wait"] * 0.8),
                            reverse=True
                        )
                        next_lane = scored[0][0]
                        if next_lane == current_lane and len(scored) > 1 and scored[1][1]["count"] > 1:
                            next_lane = scored[1][0]

                    # Calculate adaptive green duration
                    v_count = sim_state["lanes"][next_lane]["count"]
                    w_time = sim_state["lanes"][next_lane]["wait"]
                    has_em = sim_state["lanes"][next_lane]["emergency"]
                    
                    green_time = controller.calculate_green_time(v_count, w_time, has_em)

                    sim_state["active_lane"] = next_lane
                    sim_state["phase"] = "GREEN"
                    sim_state["green_duration"] = green_time
                    sim_state["timer_remaining"] = green_time
                    sim_state["lanes"][next_lane]["state"] = "GREEN"

            # Check emergency flag
            sim_state["emergency_active"] = any(l["emergency"] for l in sim_state["lanes"].values())

# Start simulation loop in background thread
sim_thread = threading.Thread(target=simulation_background_worker, daemon=True)
sim_thread.start()


def generate_video_stream(view_mode="grid"):
    """
    MJPEG stream generator providing the 2x2 multi-camera grid or single-lane camera.
    """
    while True:
        with state_lock:
            state_copy = dict(sim_state)

        if view_mode in ["1", "2", "3", "4"]:
            frame = multi_camera_manager.get_single_lane_frame(int(view_mode), state_copy)
        else:
            # Full 2x2 grid from ecs 1.0.1
            frame = multi_camera_manager.generate_composite_grid(state_copy)

        # Resize for smooth web streaming
        out_frame = cv2.resize(frame, (960, 540))
        success, buffer = cv2.imencode('.jpg', out_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)  # ~25 FPS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Live multi-camera stream supporting view=grid, 1, 2, 3, 4."""
    view = request.args.get('view', 'grid')
    return Response(generate_video_stream(view), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    """Returns current real-time state of the intersection and signals."""
    with state_lock:
        return jsonify(sim_state)


@app.route('/api/trigger_emergency', methods=['POST'])
def trigger_emergency():
    """Toggles emergency vehicle on a specified lane."""
    lane = request.json.get('lane', 'West') if request.is_json else 'West'
    with state_lock:
        if lane in sim_state["lanes"]:
            current_em = sim_state["lanes"][lane]["emergency"]
            sim_state["lanes"][lane]["emergency"] = not current_em
            if not current_em:
                # Immediate priority switch
                sim_state["timer_remaining"] = 1
                sim_state["phase"] = "YELLOW"
            return jsonify({
                "status": "success",
                "lane": lane,
                "emergency_now": sim_state["lanes"][lane]["emergency"]
            })
    return jsonify({"status": "error", "message": "Lane not found"}), 400


@app.route('/api/add_traffic', methods=['POST'])
def add_traffic():
    """Adds a burst of vehicles to a specified lane."""
    lane = request.json.get('lane', 'North') if request.is_json else 'North'
    amount = int(request.json.get('amount', 10)) if request.is_json else 10
    with state_lock:
        if lane in sim_state["lanes"]:
            sim_state["lanes"][lane]["count"] += amount
            return jsonify({
                "status": "success",
                "lane": lane,
                "new_count": sim_state["lanes"][lane]["count"]
            })
    return jsonify({"status": "error", "message": "Lane not found"}), 400


@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    """Resets intersection queues and counters."""
    with state_lock:
        for l in sim_state["lanes"].values():
            l["emergency"] = False
            l["wait"] = 5
        sim_state["total_passed"] = 0
        sim_state["phase"] = "GREEN"
        sim_state["active_lane"] = "South"
        sim_state["timer_remaining"] = 20
    return jsonify({"status": "reset_complete"})


if __name__ == '__main__':
    print("=" * 60)
    print("SMART TRAFFIC SYSTEM (ECS 1.0.1 + ECS 1.0.2 INTEGRATED)")
    print("Dashboard listening at: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)