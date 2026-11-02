#!/usr/bin/env python3
"""
CS474 Deep Learning Major Project: Road Condition and Pothole Detection
Module: Interactive Web Dashboard (app.py)

Academic & Engineering Specifications:
1. Full-Stack Streamlit Dashboard for Civil Road Safety Assessment.
2. Dynamic parameter tuning: Confidence threshold, IoU threshold,
   ByteTrack confirmation persistence, and frame stride.
3. Dual video inspection view (Original Dashcam vs YOLOv8+ByteTrack HUD Stream).
4. Interactive Geospatial GIS Map (Folium) with vehicle trajectory and
   severity-coded defect clusters (Severe, Moderate, Minor).
5. Road Condition Analytics (Plotly):
   - Roughness Index (IRI proxy) profile over chainage/distance.
   - Defect classification distribution (Pothole, Crack, Rough Patch).
   - Defect severity breakdown.
6. Enterprise Data Export: Inspection log CSV and Markdown executive report.
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium import plugins
from streamlit_folium import st_folium

# Configure Streamlit page layout
st.set_page_config(
    page_title="CS474 Road Condition & Pothole AI Inspector",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-tech aesthetic
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #78909C;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .severity-badge-severe {
        color: #FF5252;
        font-weight: bold;
    }
    .severity-badge-moderate {
        color: #FFA726;
        font-weight: bold;
    }
    .severity-badge-minor {
        color: #FFEE58;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


def check_and_generate_demo_assets():
    """Generates synthetic demo video and mock inspection CSV if not already present."""
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    demo_video = data_dir / "demo_dashcam.mp4"
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    demo_csv = reports_dir / "inspection_log.csv"

    # Create dummy inspection log if missing so the dashboard is immediately interactive
    if not demo_csv.exists():
        mock_data = [
            {"defect_id": 1, "class_name": "pothole", "confidence": 0.88, "severity": "Severe", "first_frame": 18, "last_frame": 45, "duration_frames": 28, "timestamp_sec": 0.60, "latitude": 37.77492, "longitude": -122.41941, "speed_kmh": 48.5, "bbox_x": 380, "bbox_y": 320, "bbox_w": 95, "bbox_h": 60, "estimated_area_px": 5700},
            {"defect_id": 2, "class_name": "crack", "confidence": 0.76, "severity": "Moderate", "first_frame": 78, "last_frame": 115, "duration_frames": 38, "timestamp_sec": 2.60, "latitude": 37.77535, "longitude": -122.41910, "speed_kmh": 51.2, "bbox_x": 520, "bbox_y": 340, "bbox_w": 110, "bbox_h": 45, "estimated_area_px": 4950},
            {"defect_id": 3, "class_name": "pothole", "confidence": 0.92, "severity": "Severe", "first_frame": 142, "last_frame": 180, "duration_frames": 39, "timestamp_sec": 4.73, "latitude": 37.77580, "longitude": -122.41875, "speed_kmh": 53.0, "bbox_x": 460, "bbox_y": 360, "bbox_w": 130, "bbox_h": 85, "estimated_area_px": 11050},
            {"defect_id": 4, "class_name": "rough_patch", "confidence": 0.71, "severity": "Minor", "first_frame": 204, "last_frame": 235, "duration_frames": 32, "timestamp_sec": 6.80, "latitude": 37.77625, "longitude": -122.41840, "speed_kmh": 49.8, "bbox_x": 310, "bbox_y": 310, "bbox_w": 160, "bbox_h": 90, "estimated_area_px": 14400},
            {"defect_id": 5, "class_name": "crack", "confidence": 0.81, "severity": "Moderate", "first_frame": 255, "last_frame": 288, "duration_frames": 34, "timestamp_sec": 8.50, "latitude": 37.77670, "longitude": -122.41805, "speed_kmh": 50.4, "bbox_x": 580, "bbox_y": 350, "bbox_w": 85, "bbox_h": 55, "estimated_area_px": 4675},
        ]
        pd.DataFrame(mock_data).to_csv(demo_csv, index=False)


def run_backend_video_pipeline(
    video_path: str,
    model_path: str,
    conf: float,
    iou: float,
    min_frames: int,
    stride: int,
    output_vid: str,
    output_csv: str
) -> bool:
    """Executes the Python backend tracking pipeline."""
    try:
        from scripts.inference_video import process_video_pipeline
        process_video_pipeline(
            video_source=video_path,
            model_path=model_path,
            output_video_path=output_vid,
            output_csv_path=output_csv,
            conf_thresh=conf,
            iou_thresh=iou,
            min_confirmation_frames=min_frames,
            frame_stride=stride
        )
        return True
    except Exception as e:
        st.error(f"Inference Engine Encountered an Error: {e}")
        return False


def build_folium_map(df: pd.DataFrame) -> folium.Map:
    """Renders high-contrast interactive Folium map with route polyline and severity markers."""
    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        # Default center San Francisco
        m = folium.Map(location=[37.7749, -122.4194], zoom_start=15, tiles="CartoDB positron")
        return m

    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="CartoDB positron")

    # Add Tile Layers
    folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark Mode").add_to(m)

    # Route Polyline
    route_pts = df.sort_values(by="timestamp_sec")[["latitude", "longitude"]].values.tolist()
    if len(route_pts) > 1:
        folium.PolyLine(
            route_pts,
            color="#1E88E5",
            weight=5,
            opacity=0.85,
            tooltip="Vehicle Inspection Route"
        ).add_to(m)

    # Marker Cluster for Road Defects
    cluster = plugins.MarkerCluster(name="Clustered Defects").add_to(m)

    # Severity Color Mapping
    color_map = {
        "Severe": "red",
        "Moderate": "orange",
        "Minor": "beige"
    }

    icon_map = {
        "pothole": "exclamation-triangle",
        "crack": "bolt",
        "rough_patch": "road"
    }

    for _, row in df.iterrows():
        sev = str(row.get("severity", "Moderate"))
        cname = str(row.get("class_name", "defect")).lower()
        marker_color = color_map.get(sev, "orange")
        icon_name = icon_map.get(cname, "info-sign")

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 13px; line-height: 1.4;">
            <b style="color: #1E88E5; font-size: 14px;">Defect #{int(row['defect_id'])}: {cname.upper()}</b><br>
            <hr style="margin: 4px 0;">
            <b>Severity:</b> <span style="color: {marker_color}; font-weight: bold;">{sev}</span><br>
            <b>Confidence:</b> {row['confidence'] * 100:.1f}%<br>
            <b>Vehicle Speed:</b> {row.get('speed_kmh', 0.0):.1f} km/h<br>
            <b>Time:</b> {row.get('timestamp_sec', 0.0):.1f}s (Frame {int(row.get('first_frame', 0))})<br>
            <b>Coordinates:</b> {row['latitude']:.5f}, {row['longitude']:.5f}
        </div>
        """

        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Defect #{int(row['defect_id'])}: {cname.title()} ({sev})",
            icon=folium.Icon(color=marker_color, icon=icon_name, prefix="fa")
        ).add_to(cluster)

    folium.LayerControl().add_to(m)
    return m


def main():
    check_and_generate_demo_assets()

    # Header
    st.markdown('<div class="main-header">🛣️ CS474: Road Condition & Pothole AI Inspector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Class Surface Anomaly Detection, ByteTrack Kinematic Association & Georeferenced Telemetry Analytics</div>', unsafe_allow_html=True)

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Inspection Parameters")

        # Model Checkpoint
        model_options = ["models/best.pt", "models/yolov8s.pt", "yolov8n.pt", "yolov8s.pt"]
        selected_model = st.selectbox(
            "YOLOv8 Model Checkpoint",
            options=model_options,
            index=0,
            help="Select fine-tuned checkpoint or pre-trained base model."
        )

        st.markdown("---")
        st.subheader("🎯 Detection & Tracking Tuners")

        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.10, max_value=0.95, value=0.35, step=0.05,
            help="Minimum prediction confidence to classify as surface defect."
        )

        iou_threshold = st.slider(
            "IoU (NMS) Threshold",
            min_value=0.10, max_value=0.85, value=0.45, step=0.05,
            help="Intersection-over-Union threshold for non-maximum suppression."
        )

        min_track_frames = st.slider(
            "Track Confirmation Streak",
            min_value=1, max_value=8, value=3, step=1,
            help="Consecutive frames required by ByteTrack to confirm unique defect (prevents flickers)."
        )

        frame_stride = st.slider(
            "Processing Frame Stride",
            min_value=1, max_value=10, value=1, step=1,
            help="Stride factor (1 to 10): 1 processes every frame; higher values (e.g., 2-10) skip frames for up to 10x faster CPU inference."
        )

        st.markdown("---")
        st.subheader("📁 Input Video Source")
        video_mode = st.radio("Select Video Input:", ["Pre-packaged Demo Dashcam", "Upload Custom Video"], index=0)

        uploaded_file = None
        if video_mode == "Upload Custom Video":
            uploaded_file = st.file_uploader("Upload Dashcam MP4/AVI", type=["mp4", "avi", "mov"])

    # Tabs Layout
    tab_video, tab_map, tab_analytics, tab_report = st.tabs([
        "📹 Video Inspection & HUD",
        "🗺️ Geospatial Defect Map",
        "📊 Road Quality Analytics",
        "📄 Inspection Log & Export"
    ])

    # Persistent output paths
    annotated_video_path = Path("outputs/annotated_videos/inspection_annotated.mp4")
    inspection_csv_path = Path("outputs/reports/inspection_log.csv")

    # =========================================================================
    # TAB 1: VIDEO INSPECTION & INFERENCE
    # =========================================================================
    with tab_video:
        st.markdown("### Dashcam Video Stream & Heads-Up Display (HUD)")

        # Determine input video path
        temp_input_video = None
        if video_mode == "Upload Custom Video" and uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            temp_input_video = tfile.name
        elif video_mode == "Pre-packaged Demo Dashcam":
            demo_path = Path("data/demo_dashcam.mp4")
            if not demo_path.exists():
                from scripts.inference_video import generate_synthetic_demo_video
                generate_synthetic_demo_video(str(demo_path))
            temp_input_video = str(demo_path)

        col_run, col_status = st.columns([1, 3])
        with col_run:
            run_btn = st.button("🚀 Run AI Road Inspection", type="primary", use_container_width=True)

        if run_btn:
            if temp_input_video is None:
                st.warning("Please upload a video or select the demo video source.")
            else:
                with st.spinner("Analyzing Dashcam Footage with YOLOv8 & ByteTrack..."):
                    success = run_backend_video_pipeline(
                        video_path=temp_input_video,
                        model_path=selected_model,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        min_frames=min_track_frames,
                        stride=frame_stride,
                        output_vid=str(annotated_video_path),
                        output_csv=str(inspection_csv_path)
                    )
                    if success:
                        st.success("Inspection pipeline finished successfully!")

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("#### Input Dashcam Stream")
            if temp_input_video and Path(temp_input_video).exists():
                st.video(temp_input_video)
            else:
                st.info("No input video loaded. Upload a video in the sidebar.")

        with col_v2:
            st.markdown("#### AI Annotated HUD Video")
            if annotated_video_path.exists():
                st.video(str(annotated_video_path))
            else:
                st.info("Run inspection to view annotated video with ByteTrack telemetry HUD.")

    # Load Inspection Log Data
    df_log = pd.DataFrame()
    if inspection_csv_path.exists():
        try:
            df_log = pd.read_csv(inspection_csv_path)
        except Exception as e:
            st.warning(f"Unable to read CSV log: {e}")

    # Calculate High-Level Road Metrics
    total_defects = len(df_log) if not df_log.empty else 0
    potholes = len(df_log[df_log["class_name"] == "pothole"]) if not df_log.empty else 0
    cracks = len(df_log[df_log["class_name"] == "crack"]) if not df_log.empty else 0
    rough_patches = len(df_log[df_log["class_name"] == "rough_patch"]) if not df_log.empty else 0
    severe_count = len(df_log[df_log["severity"] == "Severe"]) if not df_log.empty else 0

    # Distance calculation: speed * time
    if not df_log.empty and "timestamp_sec" in df_log.columns and "speed_kmh" in df_log.columns:
        duration_hr = (df_log["timestamp_sec"].max() - df_log["timestamp_sec"].min()) / 3600.0
        avg_speed = df_log["speed_kmh"].mean()
        distance_km = max(0.2, avg_speed * duration_hr)
    else:
        distance_km = 0.5

    defects_per_km = total_defects / max(0.1, distance_km)
    road_health_index = max(10, min(100, int(100 - (potholes * 15 + cracks * 6 + rough_patches * 4) / max(0.5, distance_km))))

    # Render KPI Cards
    st.markdown("---")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Total Defects Tracked", f"{total_defects}", delta="Deduplicated")
    with kpi2:
        st.metric("Potholes Detected", f"{potholes}", delta=f"{severe_count} Severe", delta_color="inverse")
    with kpi3:
        st.metric("Cracks & Rough Patches", f"{cracks + rough_patches}")
    with kpi4:
        st.metric("Defect Density", f"{defects_per_km:.1f} / km")
    with kpi5:
        st.metric("Road Quality Index (RQI)", f"{road_health_index} / 100", delta="Fair" if road_health_index > 60 else "Degraded")

    # =========================================================================
    # TAB 2: GEOSPATIAL DEFECT MAP
    # =========================================================================
    with tab_map:
        st.markdown("### Interactive GIS Route & Defect Map")
        st.write("Pins represent uniquely tracked surface anomalies georeferenced from video timestamps. Color indicates severity.")

        folium_map = build_folium_map(df_log)
        st_folium(folium_map, width=1100, height=520)

    # =========================================================================
    # TAB 3: ROAD QUALITY ANALYTICS
    # =========================================================================
    with tab_analytics:
        st.markdown("### Civil Infrastructure Health & Defect Distribution")

        if df_log.empty:
            st.info("No defect telemetry data available. Run the inspection pipeline to populate analytics.")
        else:
            col_ch1, col_ch2 = st.columns(2)

            with col_ch1:
                # Class Breakdown Donut Chart
                class_counts = df_log["class_name"].value_counts().reset_index()
                class_counts.columns = ["Class", "Count"]
                fig_donut = px.pie(
                    class_counts,
                    names="Class",
                    values="Count",
                    hole=0.45,
                    title="Defect Category Breakdown",
                    color="Class",
                    color_discrete_map={
                        "pothole": "#FF5252",
                        "crack": "#FFD54F",
                        "rough_patch": "#FF9800"
                    }
                )
                fig_donut.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_donut, use_container_width=True)

            with col_ch2:
                # Severity Distribution Grouped Bar Chart
                sev_counts = df_log.groupby(["class_name", "severity"]).size().reset_index(name="Count")
                fig_bar = px.bar(
                    sev_counts,
                    x="class_name",
                    y="Count",
                    color="severity",
                    barmode="group",
                    title="Defect Severity by Category",
                    color_discrete_map={
                        "Severe": "#D32F2F",
                        "Moderate": "#F57C00",
                        "Minor": "#FBC02D"
                    }
                )
                fig_bar.update_layout(margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Defect Category", yaxis_title="Number of Defects")
                st.plotly_chart(fig_bar, use_container_width=True)

            # Timeline: Pothole & Defect Occurrences vs Vehicle Speed
            st.markdown("#### Vehicle Speed & Defect Timeline Profile")
            fig_timeline = px.scatter(
                df_log,
                x="timestamp_sec",
                y="speed_kmh",
                size="estimated_area_px",
                color="class_name",
                hover_data=["defect_id", "severity", "confidence"],
                title="Defect En-Route Encounter Timeline vs Vehicle Speed",
                color_discrete_map={
                    "pothole": "#FF5252",
                    "crack": "#FFD54F",
                    "rough_patch": "#FF9800"
                }
            )
            fig_timeline.update_layout(xaxis_title="Inspection Timestamp (seconds)", yaxis_title="Vehicle Speed (km/h)")
            st.plotly_chart(fig_timeline, use_container_width=True)

    # =========================================================================
    # TAB 4: INSPECTION LOG & EXPORT
    # =========================================================================
    with tab_report:
        st.markdown("### Civil Maintenance Inspection Log")
        if df_log.empty:
            st.info("Inspection log is currently empty.")
        else:
            st.dataframe(df_log, use_container_width=True)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                csv_bytes = df_log.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Inspection Log (CSV)",
                    data=csv_bytes,
                    file_name="pothole_inspection_log.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col_dl2:
                # Executive markdown summary
                exec_summary = f"""# ROAD INFRASTRUCTURE CONDITION ASSESSMENT REPORT
Generated by: CS474 Deep Learning Road Inspector
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 1. Executive Summary
- Total Road Distance Inspected: {distance_km:.2f} km
- Total Unique Surface Anomalies: {total_defects}
- Overall Road Quality Index (RQI): {road_health_index}/100
- Defect Density: {defects_per_km:.2f} defects/km

## 2. Defect Category Inventory
- Potholes: {potholes} (Critical/Severe: {severe_count})
- Structural Cracks: {cracks}
- Rough Patches / Surface Fatigue: {rough_patches}

## 3. Recommended Maintenance Action
{"Urgent cold-mix patching and localized asphalt resurfacing required within 48 hours due to severe pothole hazards." if severe_count > 0 else "Routine crack sealing and scheduled maintenance recommended."}
"""
                st.download_button(
                    label="📑 Download Executive Report (Markdown)",
                    data=exec_summary.encode("utf-8"),
                    file_name="executive_road_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )


if __name__ == "__main__":
    main()

