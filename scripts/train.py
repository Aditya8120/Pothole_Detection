#!/usr/bin/env python3
"""
CS474 Deep Learning Major Project: Road Condition and Pothole Detection
Module: YOLOv8 Model Training & Evaluation Engine (scripts/train.py)

Academic & Technical Specifications:
1. Architecture:
   - Ultralytics YOLOv8 (Anchor-free, decoupled head architecture).
   - Dynamic checkpoint selection: yolov8n.pt (lightweight edge), yolov8s.pt (balanced),
     yolov8m.pt (high precision).
2. Loss Formulation & Justification:
   - Total Loss = lambda_box * L_CIoU + lambda_dfl * L_DFL + lambda_cls * L_BCE
   - Anchor-Free Head with Task-Aligned Assigner (TAL): Dynamically aligns classification
     and regression scores, avoiding fixed anchor box bias which severely degrades
     detection of arbitrary-aspect-ratio pavement cracks and amorphous potholes.
   - Distribution Focal Loss (DFL): Models continuous bounding box coordinate distributions
     to handle blurry road defect boundaries caused by vehicle motion.
3. Dashcam Domain-Specific Augmentation Policy:
   - fliplr=0.5 (horizontal mirror invariance).
   - flipud=0.0 (strictly disabled: preserves dashcam road perspective geometry).
   - mosaic=1.0 & mixup=0.15 (forces contextual invariance and multi-scale feature learning).
   - hsv_h=0.015, hsv_s=0.7, hsv_v=0.4 (robustness against harsh asphalt glare, shadows, overcast skies).
4. Validation & Metrics:
   - Evaluates class-wise and global Precision, Recall, mAP@0.5, and mAP@0.5:0.95.
   - Automatically exports validated weights to models/best.pt and models/yolov8s.pt.
   - Optional ONNX export with dynamic batching for edge deployment.
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TrainEngine")


def detect_optimal_device(requested_device: Optional[str] = None) -> str:
    """
    Selects the optimal hardware acceleration backend:
    CUDA (NVIDIA GPUs) -> MPS (Apple Silicon Metal) -> CPU.
    """
    if requested_device is not None and requested_device.strip() != "":
        logger.info(f"Using explicitly specified device: {requested_device}")
        return requested_device

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        logger.info(f"CUDA accelerator detected: {gpu_name} (Total GPUs: {gpu_count})")
        return "0" if gpu_count == 1 else "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple Silicon Metal Performance Shaders (MPS) detected. Accelerating on MPS.")
        return "mps"
    else:
        logger.warning("No GPU detected. Falling back to CPU. Training may be slow.")
        return "cpu"


def verify_dataset_yaml(yaml_path: Path) -> Dict[str, Any]:
    """Ensures data.yaml exists and contains required YOLOv8 fields."""
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset configuration '{yaml_path}' not found! "
            f"Run 'python scripts/setup_and_data.py' first."
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    required_keys = ["train", "val", "names"]
    for k in required_keys:
        if k not in data_cfg:
            raise ValueError(f"Malformed data.yaml: missing required field '{k}'")

    logger.info(f"Dataset verified: {data_cfg.get('nc', len(data_cfg['names']))} classes -> {data_cfg['names']}")
    return data_cfg


def train_road_detector(
    data_yaml: str = "data/data.yaml",
    model_name: str = "yolov8s.pt",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    device: Optional[str] = None,
    lr0: float = 0.01,
    optimizer: str = "AdamW",
    freeze_backbone: Optional[int] = None,
    patience: int = 15,
    workers: int = 4,
    export_onnx: bool = False,
    project_dir: str = "outputs/training_runs",
    name: str = "road_defect_yolov8"
) -> Dict[str, Any]:
    """
    Executes full YOLOv8 fine-tuning workflow with hyperparameter setup
    and validation metric persistence.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error(
            "Ultralytics package not installed! Install via:\n"
            "pip install ultralytics>=8.1.0"
        )
        sys.exit(1)

    yaml_file = Path(data_yaml).resolve()
    verify_dataset_yaml(yaml_file)

    selected_device = detect_optimal_device(device)

    # Prepare output directories
    models_dir = Path("models").resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("outputs/reports").resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("        CS474 YOLOV8 ROAD CONDITION DETECTOR TRAINING")
    logger.info("=" * 70)
    logger.info(f"Base Checkpoint : {model_name}")
    logger.info(f"Dataset YAML    : {yaml_file}")
    logger.info(f"Target Device   : {selected_device}")
    logger.info(f"Epochs          : {epochs}")
    logger.info(f"Batch Size      : {batch_size}")
    logger.info(f"Image Resolution: {img_size}x{img_size}")
    logger.info(f"Optimizer       : {optimizer} (Initial LR: {lr0})")
    logger.info(f"Early Stopping  : Patience = {patience} epochs")
    logger.info("=" * 70)

    # Initialize model from pre-trained weights
    model = YOLO(model_name)

    # Hyperparameter dictionary tailored to Dashcam Road Anomaly Detection
    train_args = {
        "data": str(yaml_file),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "device": selected_device,
        "optimizer": optimizer,
        "lr0": lr0,
        "lrf": 0.01,              # Final learning rate fraction (Cosine Annealing)
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,     # Warmup for stable convergence
        "patience": patience,     # Early stopping
        "workers": workers,
        "project": project_dir,
        "name": name,
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
        "seed": 42,

        # Domain-Specific Road Surface Augmentation Policy
        "hsv_h": 0.015,           # Subtle hue variation (simulates different asphalt compositions)
        "hsv_s": 0.7,             # Saturation variation (dry vs wet road surface)
        "hsv_v": 0.4,             # Value variation (harsh shadows, tunnel exits, dusk/dawn)
        "degrees": 0.0,           # Dashcams are rigidly mounted; rotation is minimal
        "translate": 0.1,         # Vehicle vibrations / road bumps
        "scale": 0.5,             # Multi-scale defect recognition (distant vs close defects)
        "shear": 2.0,             # Minor perspective shearing
        "perspective": 0.0005,    # Perspective warp simulating camera pitch
        "flipud": 0.0,            # Strictly 0.0: Road is always below the horizon
        "fliplr": 0.5,            # 0.5: Symmetrical road defect appearance
        "mosaic": 1.0,            # Mosaic combines 4 tiles (vital for rare defects)
        "mixup": 0.15,            # Blends images to prevent overfitting on background textures
        "copy_paste": 0.1,        # Copy-paste augmentation for small cracks
    }

    if freeze_backbone is not None and freeze_backbone > 0:
        logger.info(f"Freezing initial {freeze_backbone} backbone layers for feature reuse.")
        train_args["freeze"] = freeze_backbone

    # Train model
    results = model.train(**train_args)

    # Post-training evaluation on validation set
    logger.info("Executing comprehensive model validation on unseen test/val split...")
    val_metrics = model.val(data=str(yaml_file), imgsz=img_size, device=selected_device)

    # Extract standard metrics
    mAP50 = float(val_metrics.box.map50)
    mAP50_95 = float(val_metrics.box.map)
    precision = float(val_metrics.box.mp)
    recall = float(val_metrics.box.mr)

    logger.info("=" * 70)
    logger.info("                 VALIDATION METRICS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Precision (P)       : {precision:.4f}")
    logger.info(f"Recall (R)          : {recall:.4f}")
    logger.info(f"mAP @ IoU=0.50      : {mAP50:.4f}")
    logger.info(f"mAP @ IoU=0.50:0.95 : {mAP50_95:.4f}")
    logger.info("-" * 70)

    # Class-wise metrics
    class_metrics = {}
    if hasattr(val_metrics.box, "maps") and val_metrics.box.maps is not None:
        class_names = model.names
        for cls_idx, class_map in enumerate(val_metrics.box.maps):
            c_name = class_names.get(cls_idx, f"class_{cls_idx}")
            class_metrics[c_name] = round(float(class_map), 4)
            logger.info(f"Class [{cls_idx}] {c_name.ljust(15)} mAP50-95: {class_map:.4f}")
    logger.info("=" * 70)

    # Save metrics JSON report
    metrics_summary = {
        "model_architecture": model_name,
        "epochs_trained": epochs,
        "batch_size": batch_size,
        "image_size": img_size,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mAP_50": round(mAP50, 4),
        "mAP_50_95": round(mAP50_95, 4),
        "class_wise_mAP50_95": class_metrics
    }

    metrics_file = reports_dir / "training_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    logger.info(f"Saved validation metrics to: {metrics_file}")

    # Copy best weights to models/ directory
    run_save_dir = Path(model.trainer.save_dir)
    best_pt_path = run_save_dir / "weights" / "best.pt"
    last_pt_path = run_save_dir / "weights" / "last.pt"

    dest_best = models_dir / "best.pt"
    dest_standard = models_dir / "yolov8s.pt"

    if best_pt_path.exists():
        shutil.copy2(best_pt_path, dest_best)
        shutil.copy2(best_pt_path, dest_standard)
        logger.info(f"Exported top checkpoint -> {dest_best}")
        logger.info(f"Exported standard checkpoint -> {dest_standard}")
    elif last_pt_path.exists():
        shutil.copy2(last_pt_path, dest_best)
        shutil.copy2(last_pt_path, dest_standard)
        logger.info(f"Exported final checkpoint -> {dest_best}")

    # Optional ONNX export for inference engine acceleration
    if export_onnx:
        logger.info("Exporting model to ONNX format with dynamic axis support...")
        onnx_path = model.export(format="onnx", dynamic=True, simplify=True)
        logger.info(f"ONNX model exported successfully: {onnx_path}")
        metrics_summary["onnx_model"] = str(onnx_path)

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(
        description="CS474 YOLOv8 Road Condition & Pothole Model Training Engine"
    )
    parser.add_argument("--data", type=str, default="data/data.yaml", help="Path to data.yaml dataset config")
    parser.add_argument("--model", type=str, default="yolov8s.pt", choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], help="YOLOv8 base architecture")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for gradient updates")
    parser.add_argument("--imgsz", type=int, default=640, help="Input resolution (square pixels)")
    parser.add_argument("--device", type=str, default=None, help="Hardware device ('0', 'cpu', 'mps')")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "SGD", "Adam"], help="Optimizer")
    parser.add_argument("--freeze", type=int, default=None, help="Number of backbone layers to freeze")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--export-onnx", action="store_true", help="Export best weights to ONNX format")

    args = parser.parse_args()

    train_road_detector(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.imgsz,
        device=args.device,
        lr0=args.lr0,
        optimizer=args.optimizer,
        freeze_backbone=args.freeze,
        patience=args.patience,
        workers=args.workers,
        export_onnx=args.export_onnx
    )


if __name__ == "__main__":
    main()

