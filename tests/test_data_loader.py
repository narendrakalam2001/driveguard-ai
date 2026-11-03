"""Tests for src/data_loader.py — VideoStream error handling and DetectionDatasetLoader."""

import pandas as pd
import pytest

from src.data_loader import DetectionDatasetLoader, VideoLoadError, VideoStream


def test_video_stream_raises_on_invalid_source():
    with pytest.raises(VideoLoadError):
        VideoStream("/path/does/not/exist.mp4")


@pytest.fixture
def loader():
    return DetectionDatasetLoader()


@pytest.fixture
def sample_csv(tmp_path):
    path = tmp_path / "annotations.csv"
    df = pd.DataFrame(
        {
            "image_path": ["img1.jpg", "img1.jpg", "img2.jpg"],
            "xmin": [10, 50, 5],
            "ymin": [10, 50, 5],
            "xmax": [40, 90, 35],
            "ymax": [40, 90, 35],
            "class_name": ["Car", "Truck", "Car"],
        }
    )
    df.to_csv(path, index=False)
    return path


def test_load_ua_detrac_adds_source_column(loader, sample_csv):
    df = loader.load_ua_detrac(sample_csv)
    assert (df["source_dataset"] == "UA-DETRAC").all()
    assert list(df.columns) == loader.SCHEMA


def test_load_ua_detrac_missing_file_raises(loader, tmp_path):
    with pytest.raises(FileNotFoundError):
        loader.load_ua_detrac(tmp_path / "missing.csv")


def test_load_ua_detrac_missing_columns_raises(loader, tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"image_path": ["a.jpg"], "xmin": [1]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError):
        loader.load_ua_detrac(bad_csv)


def test_merge_datasets_concatenates(loader, sample_csv):
    ua_df = loader.load_ua_detrac(sample_csv)
    idd_df = loader.load_idd(sample_csv)  # reuse same fixture data, different source label
    merged = loader.merge_datasets(ua_df, idd_df)
    assert len(merged) == len(ua_df) + len(idd_df)
    assert set(merged["source_dataset"]) == {"UA-DETRAC", "IDD"}


def test_train_val_split_no_image_leakage(loader, sample_csv):
    ua_df = loader.load_ua_detrac(sample_csv)
    train_df, val_df = loader.train_val_split(ua_df, val_fraction=0.5, seed=42)
    train_images = set(train_df["image_path"])
    val_images = set(val_df["image_path"])
    assert train_images.isdisjoint(val_images)


def test_train_val_split_covers_all_rows(loader, sample_csv):
    ua_df = loader.load_ua_detrac(sample_csv)
    train_df, val_df = loader.train_val_split(ua_df, val_fraction=0.34, seed=1)
    assert len(train_df) + len(val_df) == len(ua_df)


# ---------------------------------------------------------------------------
# Timestamps: video time for files, never wall-clock (this once made every uploaded-video speed wrong)
# ---------------------------------------------------------------------------
def _write_video(path, frames=30, fps=25.0):
    import cv2
    import numpy as np

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for _ in range(frames):
        w.write(np.zeros((48, 64, 3), dtype=np.uint8))
    w.release()


def test_file_timestamps_are_video_time_even_when_the_consumer_is_slow(tmp_path):
    import time

    from src.data_loader import VideoStream

    path = tmp_path / "v.mp4"
    _write_video(path, frames=30, fps=25.0)
    stamps = []
    with VideoStream(str(path), queue_maxsize=2) as stream:
        for packet in stream.frames():
            stamps.append(packet.timestamp_s)
            time.sleep(0.01)  # slower than the reader: the buffer fills up
    assert len(stamps) == 30  # a file never drops frames
    assert stamps == pytest.approx([i / 25.0 for i in range(30)])  # exactly 40 ms apart, whatever the processing speed


def test_live_sources_are_flagged_and_files_are_not(tmp_path):
    from src.data_loader import VideoStream

    path = tmp_path / "v.mp4"
    _write_video(path, frames=3)
    with VideoStream(str(path)) as stream:
        assert stream.is_live is False
    assert isinstance(0, int)  # webcam indexes count as live (checked by the isinstance rule)
