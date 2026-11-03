#!/usr/bin/env python3
"""
train_plate_detector.py
========================
Fine-tunes YOLOv8n to detect license plates (ANPR stage 1) on the dataset
produced by ``scripts/prepare_plate_dataset.py``.

Usage:
    python scripts/train_plate_detector.py --dataset-dir data/processed/plate_dataset --epochs 100
    python scripts/train_plate_detector.py --imgsz 512 --time-hours 1      # fixed 1-hour budget

Works the same locally and on Kaggle (pass the Kaggle dataset folder as
--dataset-dir; the data.yaml is generated at runtime into a writable folder).

Produces:
    models/yolov8_plate.pt        best checkpoint (used by src/plate_recognizer.py)
    models/plate_model_card.json  metrics + dataset size + caveats
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import yaml

from src.config import (
    MODELS_DIR,
    PLATE_MODEL_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    YOLO_PLATE_BASE_WEIGHTS,
    get_logger,
)

logger = get_logger(__name__)

_IMG_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n as a license-plate detector.")
    parser.add_argument("--dataset-dir", type=Path, default=PROCESSED_DATA_DIR / "plate_dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--time-hours",
        type=float,
        default=None,
        help="Training time budget in hours (e.g. 1.0). Overrides --epochs: Ultralytics schedules the "
        "learning rate so training finishes inside this budget.",
    )
    parser.add_argument("--base-weights", type=str, default=YOLO_PLATE_BASE_WEIGHTS)
    parser.add_argument("--device", type=str, default="auto", help="'auto' (GPU if available else CPU), '0', or 'cpu'.")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    """Pick '0' when CUDA is available, otherwise 'cpu', unless explicitly requested."""
    if requested != "auto":
        return requested
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def count_images(folder: Path) -> int:
    """Count image files in a folder (0 if it does not exist)."""
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in _IMG_EXTS) if folder.exists() else 0


def write_data_yaml(dataset_dir: Path) -> tuple[Path, int, int]:
    """Write a data.yaml with an absolute path into the writable models dir.

    Args:
        dataset_dir: Folder containing images/{train,val} and labels/{train,val}.

    Returns:
        (path to data.yaml, n_train_images, n_val_images)

    Raises:
        SystemExit: If the folder layout is missing.
    """
    n_train = count_images(dataset_dir / "images" / "train")
    n_val = count_images(dataset_dir / "images" / "val")
    if n_train == 0 or n_val == 0:
        sys.exit(f"{dataset_dir} needs images/train and images/val with images (found {n_train} train / {n_val} val).")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = MODELS_DIR / "plate_data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {"path": str(dataset_dir.resolve()), "train": "images/train", "val": "images/val", "names": {0: "license_plate"}},
            sort_keys=False,
        )
    )
    return yaml_path, n_train, n_val


def main() -> None:
    """Train, validate, export the checkpoint and write the model card."""
    args = parse_args()
    from ultralytics import YOLO

    data_yaml, n_train, n_val = write_data_yaml(args.dataset_dir)
    device = resolve_device(args.device)
    logger.info(f"Plate detector training: {n_train} train / {n_val} val images, device={device}")

    model = YOLO(args.base_weights)
    extra = {"time": args.time_hours} if args.time_hours else {}
    if extra:
        logger.info(f"Time budget {args.time_hours} h set - --epochs is ignored.")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=device,
        seed=RANDOM_SEED,
        project=str(MODELS_DIR / "runs"),
        name="plate_detector",
        exist_ok=True,
        pretrained=True,
        verbose=True,
        **extra,
    )
    best = Path(model.trainer.best)

    metrics = YOLO(str(best)).val(
        data=str(data_yaml), device=device, project=str(MODELS_DIR / "runs"), name="plate_val", exist_ok=True
    )
    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    PLATE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, PLATE_MODEL_PATH)

    card = {
        "model_name": "DriveGuard AI License Plate Detector",
        "architecture": "YOLOv8n",
        "base_weights": args.base_weights,
        "classes": {0: "license_plate"},
        "dataset": {"train_images": n_train, "val_images": n_val, "source": str(args.dataset_dir)},
        "training": {
            "epochs": args.epochs,
            "time_hours": args.time_hours,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "seed": RANDOM_SEED,
        },
        "metrics": results,
        "caveat": (
            "Trained on a very small set of mostly close-up phone photos. Validation numbers come from "
            f"only {n_val} images and are unreliable; do not expect it to generalize to CCTV footage."
        ),
    }
    (MODELS_DIR / "plate_model_card.json").write_text(json.dumps(card, indent=2))
    logger.info(f"Saved {PLATE_MODEL_PATH} and plate_model_card.json | metrics: {results}")

    if n_train + n_val < 200:
        logger.info(f"WARNING: only {n_train + n_val} images in total - treat these metrics as a smoke test, not a result.")


if __name__ == "__main__":
    main()
