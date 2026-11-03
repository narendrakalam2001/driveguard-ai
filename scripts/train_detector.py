#!/usr/bin/env python3
"""
train_detector.py
==================
Fine-tunes YOLOv8m on the merged UA-DETRAC + IDD dataset for the
DriveGuard AI vehicle detector.

Supports TWO input shapes — mix and match freely:

1. Already-YOLO-formatted dataset directories (e.g. a Kaggle download
   that already has images/train, images/val, labels/train, labels/val):

       python scripts/train_detector.py \\
           --yolo-dirs "D:/Data_Science_Capstone_Datasets/driveguard-ai/content/UA-DETRAC/DETRAC_Upload" \\
           --epochs 60 --img-size 640 --batch 16 --device 0

   Each --yolo-dirs path must contain images/train, images/val,
   labels/train, labels/val (standard Ultralytics layout). Pass it
   multiple times to merge several such datasets:

       --yolo-dirs path/to/UA-DETRAC --yolo-dirs path/to/IDD

   IMPORTANT: class IDs across merged --yolo-dirs sources must already
   agree with VEHICLE_CLASSES in src/config.py (0=Car,1=Truck,2=Bus,
   3=Motorcycle,4=Auto-rickshaw,5=Bicycle,6=Van), OR you must supply a
   --class-map-file JSON per source (see below) — otherwise labels will
   be silently wrong. If a source ships its own data.yaml, this script
   reads it and remaps automatically.

2. Flat annotation CSVs (image_path,xmin,ymin,xmax,ymax,class_name),
   converted to YOLO format internally:

       python scripts/train_detector.py \\
           --ua-detrac-csv data/raw/ua_detrac_annotations.csv \\
           --idd-csv data/raw/idd_annotations.csv \\
           --epochs 60 --img-size 640 --batch 16 --device 0

Produces:
    models/yolov8_vehicle.pt   (best checkpoint, copied from the run dir)
    models/model_card.json     (updated with detection metrics)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict

sys.path.append(str(Path(__file__).resolve().parent.parent))

import yaml

from src.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    VEHICLE_CLASSES,
    VEHICLE_MODEL_PATH,
    YOLO_BASE_WEIGHTS,
    get_logger,
)
from src.data_loader import DetectionDatasetLoader

logger = get_logger(__name__)

_CLASS_TO_ID = {name: cid for cid, name in VEHICLE_CLASSES.items()}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training run."""
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 for DriveGuard AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Mode 1: already-YOLO-formatted directories ---
    parser.add_argument(
        "--yolo-dirs",
        type=Path,
        action="append",
        default=None,
        help="Path to a dataset already in YOLO layout (images/train, images/val, "
        "labels/train, labels/val). Repeat the flag to merge multiple datasets.",
    )
    parser.add_argument(
        "--class-map-file",
        type=Path,
        default=None,
        help="Optional JSON mapping each source's class names/ids to DriveGuard's "
        'taxonomy, keyed by --yolo-dirs path, e.g. '
        '\'{"D:/.../UA-DETRAC": {"0": "Car", "1": "Truck"}}\'. '
        "If omitted, a data.yaml inside the source dir is used if present, "
        "otherwise label class IDs are assumed to already match VEHICLE_CLASSES.",
    )

    # --- Mode 2: flat CSV annotations (legacy path) ---
    parser.add_argument("--ua-detrac-csv", type=Path, default=None, help="Flattened UA-DETRAC annotation CSV.")
    parser.add_argument("--idd-csv", type=Path, default=None, help="Flattened IDD annotation CSV.")

    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Keep every Nth image per split (e.g. 5 keeps 20%%). UA-DETRAC frames are "
        "near-duplicates, so stride 5-10 cuts training time massively with little accuracy loss.",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink images instead of copying (saves disk; needed on Kaggle's 20GB working dir).",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--base-weights", type=str, default=YOLO_BASE_WEIGHTS)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (epochs).")
    parser.add_argument("--device", type=str, default="0", help="'0' for first GPU, 'cpu' for CPU.")

    args = parser.parse_args()

    has_yolo_dirs = bool(args.yolo_dirs)
    has_csvs = bool(args.ua_detrac_csv or args.idd_csv)
    if not has_yolo_dirs and not has_csvs:
        parser.error(
            "Provide either --yolo-dirs (one or more already-YOLO-formatted "
            "dataset directories) or --ua-detrac-csv/--idd-csv (flat CSVs)."
        )
    return args


# ---------------------------------------------------------------------------
# Mode 1: merge already-YOLO-formatted directories
# ---------------------------------------------------------------------------
def _load_class_maps(args: argparse.Namespace) -> Dict[str, Dict[str, str]]:
    """Load per-source class-id -> DriveGuard-class-name maps.

    Priority per source directory: --class-map-file entry > the source's
    own data.yaml (if present) > assume label IDs already match
    VEHICLE_CLASSES (identity map).

    Args:
        args: Parsed CLI arguments.

    Returns:
        Dict keyed by source directory (str) -> {source_class_id_str: driveguard_class_name}.
    """
    external_maps: Dict[str, Dict[str, str]] = {}
    if args.class_map_file:
        external_maps = json.loads(args.class_map_file.read_text())

    resolved: Dict[str, Dict[str, str]] = {}
    for src_dir in args.yolo_dirs or []:
        key = str(src_dir)
        if key in external_maps:
            resolved[key] = external_maps[key]
            continue
        if "*" in external_maps:  # wildcard fallback: same map for any source path
            resolved[key] = external_maps["*"]
            continue

        source_yaml = src_dir / "data.yaml"
        if source_yaml.exists():
            data = yaml.safe_load(source_yaml.read_text())
            names = data.get("names", {})
            if isinstance(names, list):
                names = {str(i): n for i, n in enumerate(names)}
            resolved[key] = {str(k): v for k, v in names.items()}
            logger.info(f"Loaded class map for {src_dir} from its data.yaml: {resolved[key]}")
        else:
            # No mapping info available — assume the source's IDs already
            # match VEHICLE_CLASSES (identity map). This is the common case
            # when the data was pre-converted specifically for this project.
            resolved[key] = {str(cid): name for cid, name in VEHICLE_CLASSES.items()}
            logger.info(
                f"No data.yaml or --class-map-file entry for {src_dir}; "
                f"assuming its label IDs already match VEHICLE_CLASSES."
            )
    return resolved


def _remap_and_copy_split(
    src_images_dir: Path,
    src_labels_dir: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path,
    class_map: Dict[str, str],
    source_tag: str,
    frame_stride: int = 1,
    symlink: bool = False,
) -> int:
    """Copy one train/val split from a source YOLO dataset, remapping class IDs.

    Args:
        src_images_dir: Source images/{train,val} directory.
        src_labels_dir: Source labels/{train,val} directory.
        dst_images_dir: Destination images/{train,val} directory.
        dst_labels_dir: Destination labels/{train,val} directory.
        class_map: {source_class_id_str: driveguard_class_name}.
        source_tag: Short tag prepended to filenames to avoid collisions
            when merging multiple sources (e.g. "src0_", "src1_").
        frame_stride: Keep every Nth image (sorted by name); 1 keeps all.
        symlink: Symlink images instead of copying (falls back to copy).

    Returns:
        Number of image/label pairs copied.
    """
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    if not src_images_dir.exists():
        logger.info(f"No images dir at {src_images_dir}; skipping this split.")
        return 0

    count = 0
    image_files = sorted(
        p for p in src_images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    for idx, img_path in enumerate(image_files):
        if idx % max(1, frame_stride) != 0:
            continue
        label_path = src_labels_dir / (img_path.stem + ".txt")

        new_stem = f"{source_tag}{img_path.stem}"
        dst_img_path = dst_images_dir / f"{new_stem}{img_path.suffix}"
        dst_label_path = dst_labels_dir / f"{new_stem}.txt"

        if not dst_img_path.exists():
            if symlink:
                try:
                    dst_img_path.symlink_to(img_path.resolve())
                except OSError:
                    shutil.copy2(img_path, dst_img_path)
            else:
                shutil.copy2(img_path, dst_img_path)

        if label_path.exists():
            remapped_lines = []
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                src_class_id = parts[0]
                driveguard_name = class_map.get(src_class_id)
                if driveguard_name is None or driveguard_name not in _CLASS_TO_ID:
                    continue  # class not in our taxonomy (e.g. "person") — drop the box
                new_class_id = _CLASS_TO_ID[driveguard_name]
                remapped_lines.append(" ".join([str(new_class_id)] + parts[1:]))
            dst_label_path.write_text("\n".join(remapped_lines))
        else:
            dst_label_path.write_text("")  # background image, no objects

        count += 1
    return count


def prepare_from_yolo_dirs(args: argparse.Namespace) -> Path:
    """Merge one or more already-YOLO-formatted dataset directories.

    Args:
        args: Parsed CLI arguments (uses args.yolo_dirs).

    Returns:
        Path to the generated merged ``data.yaml``.
    """
    class_maps = _load_class_maps(args)
    yolo_root = PROCESSED_DATA_DIR / "yolo_dataset"

    total_train, total_val = 0, 0
    for i, src_dir in enumerate(args.yolo_dirs):
        class_map = class_maps[str(src_dir)]
        tag = f"src{i}_"

        n_train = _remap_and_copy_split(
            src_dir / "images" / "train",
            src_dir / "labels" / "train",
            yolo_root / "images" / "train",
            yolo_root / "labels" / "train",
            class_map,
            tag,
            args.frame_stride,
            args.symlink,
        )
        n_val = _remap_and_copy_split(
            src_dir / "images" / "val",
            src_dir / "labels" / "val",
            yolo_root / "images" / "val",
            yolo_root / "labels" / "val",
            class_map,
            tag,
            args.frame_stride,
            args.symlink,
        )
        logger.info(f"Merged {src_dir}: {n_train} train images, {n_val} val images")
        total_train += n_train
        total_val += n_val

    if total_train == 0:
        raise SystemExit(
            "No training images found across --yolo-dirs. Check that each path "
            "contains images/train and images/val subfolders with real image files."
        )

    data_yaml = {
        "path": str(yolo_root),
        "train": "images/train",
        "val": "images/val",
        "names": {cid: name for cid, name in VEHICLE_CLASSES.items()},
    }
    data_yaml_path = yolo_root / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    logger.info(
        f"Merged dataset ready at {yolo_root}: {total_train} train / {total_val} val images. "
        f"data.yaml written to {data_yaml_path}"
    )
    return data_yaml_path


# ---------------------------------------------------------------------------
# Mode 2: flat CSV annotations (legacy path, kept for datasets not
# already converted to YOLO format)
# ---------------------------------------------------------------------------
def prepare_yolo_dataset_from_csv(args: argparse.Namespace) -> Path:
    """Load, merge, split, and convert flat-CSV annotations into a YOLO-ready dataset.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Path to the generated ``data.yaml`` for Ultralytics training.
    """
    loader = DetectionDatasetLoader()
    frames = []
    if args.ua_detrac_csv:
        frames.append(loader.load_ua_detrac(args.ua_detrac_csv))
    if args.idd_csv:
        frames.append(loader.load_idd(args.idd_csv))
    merged = loader.merge_datasets(*frames) if len(frames) > 1 else frames[0]

    train_df, val_df = loader.train_val_split(merged, val_fraction=args.val_fraction, seed=RANDOM_SEED)

    yolo_root = PROCESSED_DATA_DIR / "yolo_dataset"
    train_labels_dir = yolo_root / "labels" / "train"
    val_labels_dir = yolo_root / "labels" / "val"
    train_images_dir = yolo_root / "images" / "train"
    val_images_dir = yolo_root / "images" / "val"

    for d in (train_labels_dir, val_labels_dir, train_images_dir, val_images_dir):
        d.mkdir(parents=True, exist_ok=True)

    loader.to_yolo_format(train_df, _CLASS_TO_ID, train_labels_dir)
    loader.to_yolo_format(val_df, _CLASS_TO_ID, val_labels_dir)

    for image_path in train_df["image_path"].unique():
        _link_image(Path(image_path), train_images_dir)
    for image_path in val_df["image_path"].unique():
        _link_image(Path(image_path), val_images_dir)

    data_yaml = {
        "path": str(yolo_root),
        "train": "images/train",
        "val": "images/val",
        "names": {cid: name for cid, name in VEHICLE_CLASSES.items()},
    }
    data_yaml_path = yolo_root / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    logger.info(f"YOLO dataset prepared at {yolo_root}; data.yaml written to {data_yaml_path}")
    return data_yaml_path


def _link_image(src: Path, dest_dir: Path) -> None:
    """Symlink an image into the YOLO dataset folder, falling back to copy."""
    dest = dest_dir / src.name
    if dest.exists():
        return
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dest)


# ---------------------------------------------------------------------------
# Training / evaluation (shared by both modes)
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace, data_yaml_path: Path) -> Path:
    """Run YOLOv8 fine-tuning and return the path to the best checkpoint.

    Args:
        args: Parsed CLI arguments.
        data_yaml_path: Path to the YOLO ``data.yaml``.

    Returns:
        Path to the best.pt checkpoint produced by the training run.
    """
    from ultralytics import YOLO

    model = YOLO(args.base_weights)
    model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        seed=RANDOM_SEED,
        project=str(MODELS_DIR / "runs"),
        name="vehicle_detector",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
    )
    best_ckpt = Path(model.trainer.best)
    logger.info(f"Training complete. Best checkpoint: {best_ckpt}")
    return best_ckpt


def evaluate_and_export(best_ckpt: Path) -> dict:
    """Validate the best checkpoint and copy it into the production models dir.

    Args:
        best_ckpt: Path to best.pt from the training run.

    Returns:
        Dict of validation metrics (mAP50, mAP50-95, precision, recall).
    """
    from ultralytics import YOLO

    model = YOLO(str(best_ckpt))
    metrics = model.val()

    metrics_dict = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_ckpt, VEHICLE_MODEL_PATH)
    logger.info(f"Copied best checkpoint to production path: {VEHICLE_MODEL_PATH}")
    logger.info(f"Validation metrics: {metrics_dict}")
    return metrics_dict


def write_model_card(metrics: dict, args: argparse.Namespace) -> None:
    """Write/update models/model_card.json with training metadata and metrics.

    Args:
        metrics: Validation metrics dict from :func:`evaluate_and_export`.
        args: Parsed CLI arguments (recorded for reproducibility).
    """
    card_path = MODELS_DIR / "model_card.json"
    card = {
        "model_name": "DriveGuard AI Vehicle Detector",
        "architecture": "YOLOv8m",
        "base_weights": args.base_weights,
        "classes": VEHICLE_CLASSES,
        "data_source": (
            [str(p) for p in args.yolo_dirs] if args.yolo_dirs
            else [str(p) for p in (args.ua_detrac_csv, args.idd_csv) if p]
        ),
        "training": {
            "epochs": args.epochs,
            "img_size": args.img_size,
            "batch_size": args.batch,
            "seed": RANDOM_SEED,
        },
        "metrics": metrics,
        "target_metrics": {"mAP50": 0.85, "fps_gpu": 25, "fps_cpu": 8},
    }
    card_path.write_text(json.dumps(card, indent=2))
    logger.info(f"Model card written to {card_path}")


def main() -> None:
    """Entry point: prepare dataset (either mode), train, evaluate, export."""
    args = parse_args()

    if args.yolo_dirs:
        data_yaml_path = prepare_from_yolo_dirs(args)
    else:
        data_yaml_path = prepare_yolo_dataset_from_csv(args)

    best_ckpt = train(args, data_yaml_path)
    metrics = evaluate_and_export(best_ckpt)
    write_model_card(metrics, args)

    if metrics["mAP50"] < 0.85:
        logger.info(
            f"WARNING: mAP50={metrics['mAP50']:.3f} is below the 0.85 target. "
            "Consider more epochs, more data, or hyperparameter tuning."
        )


if __name__ == "__main__":
    main()
