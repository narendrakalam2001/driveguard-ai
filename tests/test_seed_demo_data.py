"""Tests for scripts/seed_demo_data.py and scripts/frames_to_video.py."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from scripts.seed_demo_data import seed
from src.alert_system import AlertSystem

ROOT = Path(__file__).resolve().parent.parent


def test_seed_persists_only_warning_and_above(tmp_db_path):
    persisted = seed(count=60, seed_value=3, db_path=tmp_db_path)
    rows = AlertSystem(db_path=tmp_db_path).query_violations(limit=200)
    assert persisted == len(rows) > 0
    assert all(r["severity"] in {"WARNING", "VIOLATION", "SEVERE"} for r in rows)


def test_seed_is_reproducible(tmp_path):
    a = seed(count=40, seed_value=5, db_path=tmp_path / "a.db")
    b = seed(count=40, seed_value=5, db_path=tmp_path / "b.db")
    assert a == b


def _make_frames(folder: Path, numbers):
    folder.mkdir(parents=True, exist_ok=True)
    for n in numbers:
        cv2.imwrite(str(folder / f"MVI_99999_img{n:05d}.jpg"), np.zeros((64, 96, 3), dtype=np.uint8))


def test_frames_to_video_consecutive(tmp_path):
    _make_frames(tmp_path / "imgs", range(1, 21))
    out = tmp_path / "v.mp4"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "frames_to_video.py"), "--images-dir", str(tmp_path / "imgs"),
         "--sequence", "MVI_99999", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "WARNING" not in result.stdout
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 20


def test_frames_to_video_warns_on_subsampled_frames(tmp_path):
    _make_frames(tmp_path / "imgs", range(1, 201, 10))  # stride-10 like the Kaggle subset
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "frames_to_video.py"), "--images-dir", str(tmp_path / "imgs"),
         "--sequence", "MVI_99999", "--out", str(tmp_path / "v.mp4")],
        capture_output=True, text=True,
    )
    assert "not consecutive" in result.stdout
