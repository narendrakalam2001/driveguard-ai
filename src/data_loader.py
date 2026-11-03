"""
data_loader.py
===============
Loads video files / webcam streams frame-by-frame, and loads annotated
detection datasets (UA-DETRAC, Indian Driving Dataset) for training.

Two responsibilities are kept separate on purpose:
  1. VideoStream       -> runtime frame ingestion (inference time)
  2. DetectionDatasetLoader -> offline annotated-dataset ingestion (training time)
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.config import FRAME_QUEUE_MAXSIZE, RAW_DATA_DIR, get_logger

logger = get_logger(__name__)


class VideoLoadError(RuntimeError):
    """Raised when a video source cannot be opened or read."""


@dataclass
class FramePacket:
    """A single decoded frame with its metadata.

    Attributes:
        frame: BGR image array (H, W, 3).
        frame_index: Sequential index within the source stream.
        timestamp_s: Seconds elapsed since stream start.
        fps: Reported frame rate of the source.
    """

    frame: np.ndarray
    frame_index: int
    timestamp_s: float
    fps: float


class VideoStream:
    """Threaded frame reader for a video file or a live webcam index.

    Uses a background thread + bounded queue so downstream inference
    never blocks on disk/camera I/O, and drops frames gracefully under
    backpressure instead of growing memory unbounded.
    """

    def __init__(self, source: str | int, queue_maxsize: int = FRAME_QUEUE_MAXSIZE) -> None:
        """Initialize the stream.

        Args:
            source: File path (str) for a video file, or an integer
                camera index (e.g. 0) for a live webcam.
            queue_maxsize: Max frames buffered before older ones are
                dropped to keep the pipeline real-time.

        Raises:
            VideoLoadError: If the source cannot be opened.
        """
        self.source = source
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise VideoLoadError(f"Unable to open video source: {source!r}")

        # Live sources (webcam index, RTSP/HTTP URL) are stamped with the wall clock and may drop frames to stay
        # real time. Files are stamped with VIDEO time (frame_index / fps) and never drop frames: a file is read far
        # faster or slower than it plays, so wall-clock stamps would make every speed meaningless.
        self.is_live: bool = isinstance(source, int) or str(source).lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        self.fps: float = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._queue: "queue.Queue[Optional[FramePacket]]" = queue.Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        logger.info(
            f"Opened video source={source} fps={self.fps:.2f} "
            f"frames={self.frame_count} size=({self.width}x{self.height})"
        )

    def start(self) -> "VideoStream":
        """Start the background reader thread."""
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return self

    def _reader_loop(self) -> None:
        idx = 0
        t0 = time.time()
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                logger.info(f"End of stream reached for source={self.source}")
                break
            packet = FramePacket(
                frame=frame,
                frame_index=idx,
                timestamp_s=(time.time() - t0) if self.is_live else idx / self.fps,
                fps=self.fps,
            )
            if self.is_live:
                try:
                    self._queue.put(packet, timeout=1.0)
                except queue.Full:
                    # Drop oldest to keep a live pipeline real-time under backpressure.
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._queue.put_nowait(packet)
            else:
                while not self._stop_event.is_set():  # a file waits for the consumer: every frame is processed
                    try:
                        self._queue.put(packet, timeout=0.5)
                        break
                    except queue.Full:
                        continue
            idx += 1
        self._queue.put(None)  # sentinel: stream finished

    def frames(self) -> Iterator[FramePacket]:
        """Yield decoded frames until the stream ends or is stopped."""
        while True:
            packet = self._queue.get()
            if packet is None:
                break
            yield packet

    def stop(self) -> None:
        """Signal the reader thread to stop and release the capture."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._cap.release()
        logger.info(f"Stopped video source={self.source}")

    def __enter__(self) -> "VideoStream":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class DetectionDatasetLoader:
    """Loads annotated vehicle-detection datasets for training.

    Supports UA-DETRAC-style XML/CSV annotation exports and the Indian
    Driving Dataset (IDD) folder layout, normalizing both into a single
    pandas DataFrame schema:
    ``[image_path, xmin, ymin, xmax, ymax, class_name, source_dataset]``.
    """

    SCHEMA = ["image_path", "xmin", "ymin", "xmax", "ymax", "class_name", "source_dataset"]

    def __init__(self, raw_data_dir: Path = RAW_DATA_DIR) -> None:
        self.raw_data_dir = Path(raw_data_dir)

    def load_ua_detrac(self, annotations_csv: Path) -> pd.DataFrame:
        """Load UA-DETRAC annotations already converted to a flat CSV.

        Expected CSV columns: image_path, xmin, ymin, xmax, ymax, class_name.

        Args:
            annotations_csv: Path to the flattened annotation CSV.

        Returns:
            Normalized DataFrame following :attr:`SCHEMA`.

        Raises:
            FileNotFoundError: If the CSV does not exist.
            ValueError: If required columns are missing.
        """
        if not annotations_csv.exists():
            raise FileNotFoundError(f"UA-DETRAC annotation file not found: {annotations_csv}")

        df = pd.read_csv(annotations_csv)
        required = {"image_path", "xmin", "ymin", "xmax", "ymax", "class_name"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"UA-DETRAC CSV missing required columns: {missing}")

        df["source_dataset"] = "UA-DETRAC"
        logger.info(f"Loaded {len(df)} UA-DETRAC annotations from {annotations_csv}")
        return df[self.SCHEMA]

    def load_idd(self, annotations_csv: Path) -> pd.DataFrame:
        """Load Indian Driving Dataset annotations from a flattened CSV.

        Args:
            annotations_csv: Path to the flattened annotation CSV, same
                schema expectation as :meth:`load_ua_detrac`.

        Returns:
            Normalized DataFrame following :attr:`SCHEMA`.
        """
        if not annotations_csv.exists():
            raise FileNotFoundError(f"IDD annotation file not found: {annotations_csv}")

        df = pd.read_csv(annotations_csv)
        required = {"image_path", "xmin", "ymin", "xmax", "ymax", "class_name"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"IDD CSV missing required columns: {missing}")

        df["source_dataset"] = "IDD"
        logger.info(f"Loaded {len(df)} IDD annotations from {annotations_csv}")
        return df[self.SCHEMA]

    def merge_datasets(self, *frames: pd.DataFrame) -> pd.DataFrame:
        """Concatenate multiple normalized annotation frames.

        Args:
            *frames: Any number of DataFrames matching :attr:`SCHEMA`.

        Returns:
            A single concatenated, index-reset DataFrame.
        """
        merged = pd.concat(frames, ignore_index=True)
        logger.info(f"Merged dataset size: {len(merged)} annotated boxes")
        return merged

    def train_val_split(
        self, df: pd.DataFrame, val_fraction: float = 0.15, seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split by unique image_path so no image leaks across splits.

        Args:
            df: Annotation DataFrame following :attr:`SCHEMA`.
            val_fraction: Fraction of unique images held out for validation.
            seed: Random seed for reproducibility.

        Returns:
            (train_df, val_df) tuple.
        """
        rng = np.random.default_rng(seed)
        unique_images = np.array(df["image_path"].unique(), dtype=object)
        rng.shuffle(unique_images)
        n_val = max(1, int(len(unique_images) * val_fraction))
        val_images = set(unique_images[:n_val])

        val_df = df[df["image_path"].isin(val_images)].reset_index(drop=True)
        train_df = df[~df["image_path"].isin(val_images)].reset_index(drop=True)
        logger.info(
            f"Split into train={len(train_df)} boxes / val={len(val_df)} boxes "
            f"({len(unique_images) - n_val} / {n_val} images)"
        )
        return train_df, val_df

    def to_yolo_format(
        self, df: pd.DataFrame, class_to_id: dict, output_dir: Path, image_size_lookup=None
    ) -> None:
        """Convert normalized annotations into YOLO .txt label files.

        Args:
            df: Annotation DataFrame following :attr:`SCHEMA`.
            class_to_id: Mapping from class_name -> integer class id.
            output_dir: Directory where per-image .txt label files go.
            image_size_lookup: Optional callable(image_path) -> (w, h).
                If omitted, image dimensions are read with OpenCV.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for image_path, group in df.groupby("image_path"):
            if image_size_lookup is not None:
                w, h = image_size_lookup(image_path)
            else:
                img = cv2.imread(str(image_path))
                if img is None:
                    logger.info(f"Skipping unreadable image: {image_path}")
                    continue
                h, w = img.shape[:2]

            lines: List[str] = []
            for _, row in group.iterrows():
                if row["class_name"] not in class_to_id:
                    continue
                cls_id = class_to_id[row["class_name"]]
                x_center = ((row["xmin"] + row["xmax"]) / 2) / w
                y_center = ((row["ymin"] + row["ymax"]) / 2) / h
                box_w = (row["xmax"] - row["xmin"]) / w
                box_h = (row["ymax"] - row["ymin"]) / h
                lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

            label_path = output_dir / (Path(image_path).stem + ".txt")
            label_path.write_text("\n".join(lines))

        logger.info(f"Wrote YOLO label files to {output_dir}")
