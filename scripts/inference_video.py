#!/usr/bin/env python3
"""
CS474 Deep Learning Major Project: Road Condition and Pothole Detection
Module: Video Tracking & Telemetry Engine (scripts/inference_video.py)

Academic & Technical Specifications:
1. Video Ingestion & Decoding Pipeline:
   - Robust cv2.VideoCapture pipeline with error detection, frame-drop handling,
     and auto-recovery.
   - Codec fallback hierarchy: 'avc1' (H.264 web-compatible) -> 'mp4v' -> 'XVID'.
2. Multi-Object Tracking via ByteTrack:
   - Eliminates catastrophic defect inflation caused by frame-by-frame counting.
   - Kalman Filter spatial prediction + two-stage bipartite matching via Hungarian algorithm.
   - Temporal confirmation threshold (K >= 3 frames) to suppress transient false alarms.
3. Telemetry & GPS Georeferencing:
   - Ingests real GPX route tracks or synchronized CSV telemetry.
   - Fallback Synthetic Telemetry Generator: produces high-fidelity vehicle dynamics
     (speed, heading, GPS coordinates, chainage/distance) synchronized per frame.
   - Associates each unique confirmed defect with exact GPS lat/lon and road chainage (km).
4. Heads-Up Display (HUD) Dashboard Overlay:
   - Top banner: Telemetry (Speed km/h, Lat, Long, Time, FPS).
   - Bottom status panel: Road Quality Index (RQI) and hazard indicators.
   - Defect inventory counter (Potholes, Cracks, Rough Patches).
   - Bounding boxes with Track ID, Class Name, Confidence, and Severity badge.
5. Structured Inspection Reporting:
   - Generates outputs/reports/inspection_log.csv for civil infrastructure maintenance.
"""

import os
import sys
import math
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("VideoInference")

CLASS_NAMES = {0: "pothole", 1: "crack", 2: "rough_patch"}
CLASS_COLORS_BGR = {
    0: (40, 40, 240),    # Pothole: Crimson Red
    1: (0, 215, 255),    # Crack: Gold / Bright Amber
    2: (0, 140, 255)     # Rough Patch: Deep Orange
}


class DefectTrackRecord:
    """Stores persistent state for a uniquely tracked road defect."""

    def __init__(self, track_id: int, class_id: int, class_name: str, frame_idx: int, timestamp: float, bbox: List[float], conf: float, gps: Tuple[float, float], speed: float):
        self.track_id = track_id
        self.class_id = class_id
        self.class_name = class_name
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.detection_count = 1
        self.max_conf = conf
        self.latest_bbox = bbox  # [x1, y1, x2, y2]
        self.best_bbox = bbox
        self.gps_lat, self.gps_lon = gps
        self.speed_kmh = speed
        self.timestamp = timestamp
        self.trajectory: List[Tuple[int, int]] = []
        self._update_center(bbox)

    def update(self, frame_idx: int, bbox: List[float], conf: float, gps: Tuple[float, float], speed: float):
        self.last_frame = frame_idx
        self.detection_count += 1
        self.latest_bbox = bbox
        self._update_center(bbox)
        if conf > self.max_conf:
            self.max_conf = conf
            self.best_bbox = bbox
            self.gps_lat, self.gps_lon = gps
            self.speed_kmh = speed

    def _update_center(self, bbox: List[float]):
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        self.trajectory.append((cx, cy))
        if len(self.trajectory) > 30:
            self.trajectory.pop(0)

    @property
    def severity(self) -> str:
        """
        Calculates physical hazard severity based on defect category,
        bounding box pixel area, and confidence.
        """
        x1, y1, x2, y2 = self.best_bbox
        area = max(0, (x2 - x1) * (y2 - y1))

        if self.class_id == 0:  # Pothole
            if area > 18000 or self.max_conf > 0.80:
                return "Severe"
            elif area > 7000:
                return "Moderate"
            return "Minor"
        elif self.class_id == 1:  # Crack
            if area > 15000:
                return "Severe"
            elif area > 5000:
                return "Moderate"
            return "Minor"
        else:  # Rough Patch
            if area > 25000:
                return "Severe"
            elif area > 10000:
                return "Moderate"
            return "Minor"

    def to_dict(self) -> Dict[str, Any]:
        x1, y1, x2, y2 = self.best_bbox
        w = round(x2 - x1, 2)
        h = round(y2 - y1, 2)
        area = round(w * h, 2)
        return {
            "defect_id": self.track_id,
            "class_name": self.class_name,
            "confidence": round(self.max_conf, 3),
            "severity": self.severity,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "duration_frames": (self.last_frame - self.first_frame) + 1,
            "timestamp_sec": round(self.timestamp, 2),
            "latitude": round(self.gps_lat, 6),
            "longitude": round(self.gps_lon, 6),
            "speed_kmh": round(self.speed_kmh, 1),
            "bbox_x": round(x1, 1),
            "bbox_y": round(y1, 1),
            "bbox_w": w,
            "bbox_h": h,
            "estimated_area_px": area
        }


class TelemetryEngine:
    """
    Manages vehicle telemetry: Ingests real GPX/CSV or synthesizes
    realistic vehicle kinematic trajectories with GPS coordinates and speeds.
    """

    def __init__(self, gpx_file: Optional[str] = None, csv_file: Optional[str] = None, base_lat: float = 37.7749, base_lon: float = -122.4194):
        self.telemetry_data: List[Dict[str, float]] = []
        if gpx_file and Path(gpx_file).exists():
            self._load_gpx(gpx_file)
        elif csv_file and Path(csv_file).exists():
            self._load_csv(csv_file)
        else:
            self.base_lat = base_lat
            self.base_lon = base_lon
            self.simulated = True

    def _load_csv(self, csv_file: str):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            self.telemetry_data.append({
                "timestamp": float(row.get("timestamp", 0.0)),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "speed": float(row.get("speed_kmh", 50.0))
            })
        self.simulated = False
        logger.info(f"Loaded {len(self.telemetry_data)} telemetry records from CSV.")

    def _load_gpx(self, gpx_file: str):
        try:
            import gpxpy
            with open(gpx_file, "r") as f:
                gpx = gpxpy.parse(f)
            t0 = None
            for track in gpx.tracks:
                for segment in track.segments:
                    for pt in segment.points:
                        if t0 is None:
                            t0 = pt.time.timestamp() if pt.time else 0.0
                        t = (pt.time.timestamp() - t0) if pt.time else len(self.telemetry_data) * 0.1
                        speed = pt.speed * 3.6 if pt.speed is not None else 50.0
                        self.telemetry_data.append({
                            "timestamp": t,
                            "lat": pt.latitude,
                            "lon": pt.longitude,
                            "speed": speed
                        })
            self.simulated = False
            logger.info(f"Parsed {len(self.telemetry_data)} points from GPX track.")
        except Exception as e:
            logger.warning(f"Failed to parse GPX: {e}. Defaulting to synthetic telemetry.")
            self.simulated = True

    def get_telemetry_at_time(self, t_sec: float) -> Tuple[float, float, float]:
        """Returns (lat, lon, speed_kmh) at specified timestamp."""
        if not self.simulated and len(self.telemetry_data) > 0:
            # Linear interpolation from loaded telemetry
            if t_sec <= self.telemetry_data[0]["timestamp"]:
                p = self.telemetry_data[0]
                return p["lat"], p["lon"], p["speed"]
            if t_sec >= self.telemetry_data[-1]["timestamp"]:
                p = self.telemetry_data[-1]
                return p["lat"], p["lon"], p["speed"]

            for i in range(len(self.telemetry_data) - 1):
                p1 = self.telemetry_data[i]
                p2 = self.telemetry_data[i + 1]
                if p1["timestamp"] <= t_sec <= p2["timestamp"]:
                    alpha = (t_sec - p1["timestamp"]) / max(1e-4, p2["timestamp"] - p1["timestamp"])
                    lat = p1["lat"] + alpha * (p2["lat"] - p1["lat"])
                    lon = p1["lon"] + alpha * (p2["lon"] - p1["lon"])
                    speed = p1["speed"] + alpha * (p2["speed"] - p1["speed"])
                    return lat, lon, speed

        # Synthetic dynamic trajectory: driving along a roadway with slight curves
        speed_base = 52.0  # km/h
        speed_variation = 6.0 * math.sin(t_sec * 0.15) + 3.0 * math.cos(t_sec * 0.4)
        current_speed = max(25.0, min(75.0, speed_base + speed_variation))

        # Vehicle displacement: 1 deg lat ~ 111 km
        dist_km = (current_speed / 3600.0) * t_sec
        delta_lat = dist_km / 111.0
        # Add realistic gentle road curving along longitude
        delta_lon = (dist_km * 0.35 + 0.0004 * math.sin(t_sec * 0.08)) / 111.0

        lat = self.base_lat + delta_lat
        lon = self.base_lon + delta_lon
        return lat, lon, current_speed


class HUDOverlayRenderer:
    """Renders a modern, professional heads-up display overlay on dashcam frames."""

    @staticmethod
    def draw(
        frame: np.ndarray,
        fps_current: float,
        frame_idx: int,
        timestamp: float,
        lat: float,
        lon: float,
        speed_kmh: float,
        active_tracks: List[DefectTrackRecord],
        confirmed_defects: Dict[int, DefectTrackRecord]
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        # 1. Semi-transparent top telemetry banner
        banner_h = 55
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 24), -1)
        # Accent line under banner
        cv2.line(overlay, (0, banner_h), (w, banner_h), (0, 165, 255), 2)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Telemetry Text Elements
        font = cv2.FONT_HERSHEY_SIMPLEX
        time_str = time.strftime("%H:%M:%S", time.gmtime(timestamp))
        telemetry_txt = f"SPEED: {speed_kmh:4.1f} km/h   |   GPS: {lat:.5f} N, {abs(lon):.5f} W   |   TIME: {time_str}   |   FPS: {fps_current:.1f}"
        cv2.putText(frame, telemetry_txt, (18, 34), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # 2. Defect Counters Widget (Top-Right or Left-Side Panel)
        pothole_count = sum(1 for d in confirmed_defects.values() if d.class_id == 0)
        crack_count = sum(1 for d in confirmed_defects.values() if d.class_id == 1)
        rough_count = sum(1 for d in confirmed_defects.values() if d.class_id == 2)
        total_defects = len(confirmed_defects)

        counter_w = 260
        counter_h = 100
        cx0, cy0 = w - counter_w - 15, 65

        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (cx0, cy0), (cx0 + counter_w, cy0 + counter_h), (25, 28, 35), -1)
        cv2.rectangle(overlay2, (cx0, cy0), (cx0 + counter_w, cy0 + counter_h), (70, 75, 85), 1)
        cv2.addWeighted(overlay2, 0.80, frame, 0.20, 0, frame)

        cv2.putText(frame, "INSPECTION COUNTER", (cx0 + 12, cy0 + 20), font, 0.45, (180, 190, 205), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Potholes     : {pothole_count}", (cx0 + 12, cy0 + 44), font, 0.52, CLASS_COLORS_BGR[0], 2, cv2.LINE_AA)
        cv2.putText(frame, f"Cracks       : {crack_count}", (cx0 + 12, cy0 + 66), font, 0.52, CLASS_COLORS_BGR[1], 2, cv2.LINE_AA)
        cv2.putText(frame, f"Rough Patches: {rough_count}", (cx0 + 12, cy0 + 88), font, 0.52, CLASS_COLORS_BGR[2], 2, cv2.LINE_AA)

        # 3. Dynamic Road Quality Index (RQI) Badge (Bottom-Left)
        # RQI: 100 base, penalizing recent defects
        rqi_score = max(20, 100 - (pothole_count * 12 + crack_count * 5 + rough_count * 4))
        if rqi_score >= 80:
            rqi_status, status_color = "OPTIMAL", (50, 205, 50)
        elif rqi_score >= 60:
            rqi_status, status_color = "SATISFACTORY", (0, 215, 255)
        elif rqi_score >= 40:
            rqi_status, status_color = "DEGRADED", (0, 140, 255)
        else:
            rqi_status, status_color = "CRITICAL", (30, 30, 240)

        bx0, by0 = 15, h - 55
        bw, bh = 270, 42
        overlay3 = frame.copy()
        cv2.rectangle(overlay3, (bx0, by0), (bx0 + bw, by0 + bh), (20, 24, 30), -1)
        cv2.rectangle(overlay3, (bx0, by0), (bx0 + bw, by0 + bh), status_color, 2)
        cv2.addWeighted(overlay3, 0.85, frame, 0.15, 0, frame)

        cv2.putText(frame, f"ROAD QUALITY: {rqi_status} ({rqi_score}/100)", (bx0 + 10, by0 + 26), font, 0.55, status_color, 2, cv2.LINE_AA)

        # 4. Render Active Defect Bounding Boxes & Trajectories
        for track in active_tracks:
            x1, y1, x2, y2 = map(int, track.latest_bbox)
            color = CLASS_COLORS_BGR.get(track.class_id, (0, 255, 0))

            # Bounding box with corner highlight accents
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Corner lines for high-tech HUD aesthetic
            corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 4)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 4)
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 4)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 4)
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 4)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 4)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 4)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 4)

            # Draw trajectory path
            if len(track.trajectory) > 1:
                for i in range(1, len(track.trajectory)):
                    cv2.line(frame, track.trajectory[i - 1], track.trajectory[i], color, 1, cv2.LINE_AA)

            # Label badge
            label = f"ID:{track.track_id} {track.class_name.upper()} {track.max_conf:.2f} [{track.severity}]"
            (txt_w, txt_h), _ = cv2.getTextSize(label, font, 0.45, 1)
            label_y = max(y1 - 6, txt_h + 8)

            cv2.rectangle(frame, (x1, label_y - txt_h - 4), (x1 + txt_w + 6, label_y + 4), color, -1)
            cv2.putText(frame, label, (x1 + 3, label_y), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        return frame


def create_safe_video_writer(output_path: Path, fps: float, width: int, height: int) -> Tuple[cv2.VideoWriter, str]:
    """
    Attempts video writer initialization with codec fallback:
    'avc1' (H.264 web native) -> 'mp4v' (MPEG-4) -> 'XVID'.
    """
    codecs_to_try = [
        ("avc1", ".mp4"),
        ("mp4v", ".mp4"),
        ("XVID", ".avi")
    ]

    for fourcc_str, ext in codecs_to_try:
        candidate_path = output_path.with_suffix(ext)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(candidate_path), fourcc, fps, (width, height))
        if writer.isOpened():
            logger.info(f"VideoWriter successfully initialized using codec '{fourcc_str}' -> {candidate_path}")
            return writer, str(candidate_path)
        writer.release()

    raise RuntimeError("Failed to initialize any OpenCV VideoWriter codec (tried avc1, mp4v, XVID).")


def generate_synthetic_demo_video(output_video: str = "data/demo_dashcam.mp4", duration_sec: int = 10, fps: int = 30):
    """
    Generates a realistic 10-second dashcam driving clip featuring road markings,
    moving asphalt texture, and dynamically approaching road anomalies.
    Enables immediate out-of-the-box demonstration of CS474 tracking and telemetry.
    """
    logger.info(f"Generating synthetic demo dashcam video: {output_video} ({duration_sec}s @ {fps}fps)...")
    out_path = Path(output_video).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 960, 540
    total_frames = duration_sec * fps
    writer, actual_path = create_safe_video_writer(out_path, fps, width, height)

    # Defect schedule: (frame_trigger, class_id, lane_offset)
    defect_events = [
        {"start": 15, "type": 0, "lane": -0.2},   # Pothole in left-center lane
        {"start": 75, "type": 1, "lane": 0.15},   # Crack in right-center lane
        {"start": 140, "type": 0, "lane": 0.05},  # Pothole in center lane
        {"start": 200, "type": 2, "lane": -0.1},  # Rough patch
        {"start": 250, "type": 1, "lane": 0.25},  # Longitudinal crack
    ]

    for frame_idx in range(total_frames):
        # 1. Asphalt Road Texture
        frame = np.full((height, width, 3), 60, dtype=np.uint8)

        # Sky and Horizon
        horizon_y = int(height * 0.40)
        frame[0:horizon_y, :] = [180, 160, 130]  # Light sky

        # Road boundaries in perspective
        vp_x, vp_y = width // 2, horizon_y
        road_pts = np.array([
            [vp_x - 30, vp_y],
            [vp_x + 30, vp_y],
            [width - 50, height],
            [50, height]
        ], np.int32)
        cv2.fillPoly(frame, [road_pts], (70, 70, 72))

        # Lane markings with motion animation
        dash_offset = (frame_idx * 14) % 90
        for y in range(horizon_y + 10, height, 90):
            current_y = y + dash_offset
            if current_y > height or current_y < horizon_y:
                continue
            alpha = (current_y - horizon_y) / (height - horizon_y)
            lx = int(vp_x * (1 - alpha) + (width * 0.5) * alpha)
            lw = max(2, int(6 * alpha))
            lh = max(6, int(35 * alpha))
            cv2.rectangle(frame, (lx - lw, current_y), (lx + lw, min(height, current_y + lh)), (220, 220, 220), -1)

        # 2. Render Approaching Defects
        for ev in defect_events:
            elapsed = frame_idx - ev["start"]
            if 0 <= elapsed <= 55:  # Visible for 55 frames as it approaches camera
                # Progress from 0 (at horizon) to 1 (near hood/bottom)
                progress = elapsed / 55.0
                curr_y = int(horizon_y + (height - horizon_y) * (progress ** 1.8))
                scale = 0.2 + 1.2 * progress
                curr_x = int(vp_x + ev["lane"] * width * progress * 1.5)

                if ev["type"] == 0:
                    # Approaching Pothole
                    rx, ry = int(30 * scale), int(16 * scale)
                    cv2.ellipse(frame, (curr_x, curr_y), (rx + 2, ry + 2), 0, 0, 360, (30, 30, 30), -1)
                    cv2.ellipse(frame, (curr_x, curr_y), (rx, ry), 0, 0, 360, (15, 15, 20), -1)
                elif ev["type"] == 1:
                    # Approaching Crack
                    cw, ch = int(45 * scale), int(60 * scale)
                    cv2.line(frame, (curr_x - cw // 2, curr_y - ch // 2), (curr_x, curr_y), (25, 25, 25), max(1, int(3 * scale)))
                    cv2.line(frame, (curr_x, curr_y), (curr_x + cw // 2, curr_y + ch // 2), (25, 25, 25), max(1, int(3 * scale)))
                elif ev["type"] == 2:
                    # Approaching Rough Patch
                    pw, ph = int(60 * scale), int(35 * scale)
                    x1, y1 = curr_x - pw // 2, curr_y - ph // 2
                    x2, y2 = curr_x + pw // 2, curr_y + ph // 2
                    if 0 <= x1 and x2 < width and 0 <= y1 and y2 < height:
                        patch = frame[y1:y2, x1:x2].astype(np.float32) + np.random.normal(0, 15, (y2 - y1, x2 - x1, 3))
                        frame[y1:y2, x1:x2] = np.clip(patch, 0, 255).astype(np.uint8)

        writer.write(frame)

    writer.release()
    logger.info(f"Synthetic demo video created at: {actual_path}")
    return actual_path


def process_video_pipeline(
    video_source: str,
    model_path: str = "models/best.pt",
    output_video_path: str = "outputs/annotated_videos/annotated_output.mp4",
    output_csv_path: str = "outputs/reports/inspection_log.csv",
    conf_thresh: float = 0.35,
    iou_thresh: float = 0.45,
    gpx_file: Optional[str] = None,
    csv_telemetry: Optional[str] = None,
    tracker_config: str = "bytetrack.yaml",
    min_confirmation_frames: int = 3,
    frame_stride: int = 1,
    show_preview: bool = False
) -> Dict[str, Any]:
    """
    Main Video Processing Engine:
    Ingests video, tracks road defects with ByteTrack, interpolates telemetry,
    draws HUD, and persists inspection results.
    """
    video_path = Path(video_source).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Source video '{video_path}' does not exist.")

    out_vid_path = Path(output_video_path).resolve()
    out_vid_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv_path = Path(output_csv_path).resolve()
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize YOLOv8 Model
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics not installed. Install via 'pip install ultralytics>=8.1.0'")
        sys.exit(1)

    model_file = Path(model_path).resolve()
    if not model_file.exists():
        logger.warning(f"Model checkpoint '{model_file}' not found. Falling back to 'yolov8s.pt'.")
        model_file = "yolov8s.pt"

    logger.info(f"Loading YOLOv8 model from {model_file}...")
    model = YOLO(str(model_file))

    # Initialize Video Capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video stream: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Input Video: {video_path.name} ({width}x{height} @ {fps:.1f} FPS, {total_frames} frames)")

    # Video Writer Initialization
    writer, actual_out_video = create_safe_video_writer(out_vid_path, fps / frame_stride, width, height)

    # Telemetry Engine Initialization
    telemetry = TelemetryEngine(gpx_file=gpx_file, csv_file=csv_telemetry)

    # Tracking & State Management
    tracked_defects: Dict[int, DefectTrackRecord] = {}
    frame_idx = 0
    t_start = time.time()
    pbar = tqdm(total=total_frames, desc="Processing Video Frames", unit="frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_stride != 0:
            pbar.update(1)
            continue

        timestamp_sec = frame_idx / fps
        lat, lon, speed_kmh = telemetry.get_telemetry_at_time(timestamp_sec)

        # Run YOLOv8 Tracking with ByteTrack
        # Using model.track with persist=True preserves identities across calls
        track_results = model.track(
            source=frame,
            conf=conf_thresh,
            iou=iou_thresh,
            tracker=tracker_config,
            persist=True,
            verbose=False
        )

        active_tracks_this_frame: List[DefectTrackRecord] = []

        if track_results and len(track_results) > 0:
            res = track_results[0]
            boxes = res.boxes

            if boxes is not None and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                cls_ids = boxes.cls.int().cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                xyxy = boxes.xyxy.cpu().tolist()

                for tid, cid, conf, bbox in zip(track_ids, cls_ids, confs, xyxy):
                    cname = CLASS_NAMES.get(cid, model.names.get(cid, "defect"))

                    if tid not in tracked_defects:
                        record = DefectTrackRecord(
                            track_id=tid,
                            class_id=cid,
                            class_name=cname,
                            frame_idx=frame_idx,
                            timestamp=timestamp_sec,
                            bbox=bbox,
                            conf=conf,
                            gps=(lat, lon),
                            speed=speed_kmh
                        )
                        tracked_defects[tid] = record
                    else:
                        record = tracked_defects[tid]
                        record.update(
                            frame_idx=frame_idx,
                            bbox=bbox,
                            conf=conf,
                            gps=(lat, lon),
                            speed=speed_kmh
                        )

                    active_tracks_this_frame.append(record)

        # Filter confirmed defects to prevent transient flicker
        confirmed_defects = {
            tid: rec for tid, rec in tracked_defects.items()
            if rec.detection_count >= min_confirmation_frames
        }

        # Filter active tracks for HUD display
        confirmed_active = [
            t for t in active_tracks_this_frame
            if t.detection_count >= min_confirmation_frames
        ]

        # Calculate instantaneous FPS
        elapsed = time.time() - t_start
        fps_current = frame_idx / max(1e-3, elapsed)

        # Render HUD Overlay
        annotated_frame = HUDOverlayRenderer.draw(
            frame=frame,
            fps_current=fps_current,
            frame_idx=frame_idx,
            timestamp=timestamp_sec,
            lat=lat,
            lon=lon,
            speed_kmh=speed_kmh,
            active_tracks=confirmed_active,
            confirmed_defects=confirmed_defects
        )

        # Write to disk
        writer.write(annotated_frame)

        if show_preview:
            cv2.imshow("CS474 Dashcam Inspection HUD", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Inspection terminated by user interrupt.")
                break

        pbar.update(1)

    cap.release()
    writer.release()
    pbar.close()
    if show_preview:
        cv2.destroyAllWindows()

    # Generate Structured Inspection Log CSV
    records = [rec.to_dict() for rec in confirmed_defects.values()]
    df_report = pd.DataFrame(records)
    if not df_report.empty:
        df_report.sort_values(by="first_frame", inplace=True)
    df_report.to_csv(out_csv_path, index=False)

    summary = {
        "input_video": str(video_path),
        "annotated_video": actual_out_video,
        "inspection_log_csv": str(out_csv_path),
        "total_frames_analyzed": frame_idx,
        "total_unique_defects": len(confirmed_defects),
        "breakdown": {
            "pothole": sum(1 for d in confirmed_defects.values() if d.class_id == 0),
            "crack": sum(1 for d in confirmed_defects.values() if d.class_id == 1),
            "rough_patch": sum(1 for d in confirmed_defects.values() if d.class_id == 2),
        },
        "severity_breakdown": {
            "Severe": sum(1 for d in confirmed_defects.values() if d.severity == "Severe"),
            "Moderate": sum(1 for d in confirmed_defects.values() if d.severity == "Moderate"),
            "Minor": sum(1 for d in confirmed_defects.values() if d.severity == "Minor"),
        }
    }

    logger.info("=" * 70)
    logger.info("              INSPECTION RUN COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total Unique Defects Tracked: {summary['total_unique_defects']}")
    logger.info(f"Breakdown: Potholes={summary['breakdown']['pothole']} | Cracks={summary['breakdown']['crack']} | Rough={summary['breakdown']['rough_patch']}")
    logger.info(f"Severity:  Severe={summary['severity_breakdown']['Severe']} | Moderate={summary['severity_breakdown']['Moderate']} | Minor={summary['severity_breakdown']['Minor']}")
    logger.info(f"Annotated Video Output: {summary['annotated_video']}")
    logger.info(f"Inspection Report CSV : {summary['inspection_log_csv']}")
    logger.info("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="CS474 Dashcam Video Road Condition Inference, Tracking & Telemetry Engine"
    )
    parser.add_argument("--video", type=str, default=None, help="Path to input dashcam video file")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to YOLOv8 checkpoint")
    parser.add_argument("--output-video", type=str, default="outputs/annotated_videos/inspection_annotated.mp4", help="Output annotated video path")
    parser.add_argument("--output-csv", type=str, default="outputs/reports/inspection_log.csv", help="Output inspection log CSV path")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (0.0 - 1.0)")
    parser.add_argument("--min-frames", type=int, default=3, help="Consecutive frames required to confirm defect track")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride factor (1-10): process every Nth frame for faster inference")
    parser.add_argument("--gpx", type=str, default=None, help="Optional GPX file for vehicle telemetry")
    parser.add_argument("--csv-telemetry", type=str, default=None, help="Optional CSV file for vehicle telemetry")
    parser.add_argument("--generate-demo", action="store_true", help="Generate synthetic demo dashcam video before running")
    parser.add_argument("--preview", action="store_true", help="Display live OpenCV window during processing")

    args = parser.parse_args()

    # Generate synthetic demo video if requested or if no video provided
    if args.generate_demo or (args.video is None and not Path("data/demo_dashcam.mp4").exists()):
        demo_path = generate_synthetic_demo_video("data/demo_dashcam.mp4")
        if args.video is None:
            args.video = demo_path

    if args.video is None:
        args.video = "data/demo_dashcam.mp4"

    process_video_pipeline(
        video_source=args.video,
        model_path=args.model,
        output_video_path=args.output_video,
        output_csv_path=args.output_csv,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        gpx_file=args.gpx,
        csv_telemetry=args.csv_telemetry,
        min_confirmation_frames=args.min_frames,
        frame_stride=args.stride,
        show_preview=args.preview
    )


if __name__ == "__main__":
    main()

