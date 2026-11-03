"""Tests for scripts/prepare_plate_dataset.py - VOC -> YOLO conversion and safety checks."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.prepare_plate_dataset import parse_voc, prepare, to_yolo_line

VOC = """<annotation><filename>{n}.jpg</filename>
<size><width>{w}</width><height>{h}</height><depth></depth></size>
<object><name>number_plate</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>
<object><name>car</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object>
</annotation>"""


def _write(folder_img: Path, folder_xml: Path, name: str, img_w=400, img_h=200, xml_w=None, xml_h=None, box=(100, 50, 300, 150)):
    folder_img.mkdir(parents=True, exist_ok=True)
    folder_xml.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder_img / f"{name}.jpg"), np.zeros((img_h, img_w, 3), dtype=np.uint8))
    (folder_xml / f"{name}.xml").write_text(
        VOC.format(n=name, w=xml_w or img_w, h=xml_h or img_h, x1=box[0], y1=box[1], x2=box[2], y2=box[3])
    )


def test_to_yolo_line_math():
    assert to_yolo_line((100, 50, 300, 150), 400, 200) == "0 0.500000 0.500000 0.500000 0.500000"


def test_to_yolo_line_clips_and_rejects_degenerate():
    assert to_yolo_line((-20, -20, 40, 40), 100, 100) == "0 0.200000 0.200000 0.400000 0.400000"
    assert to_yolo_line((10, 10, 10.5, 50), 100, 100) is None


def test_parse_voc_keeps_only_plate_objects(tmp_path):
    _write(tmp_path / "i", tmp_path / "a", "img1")
    w, h, boxes = parse_voc(tmp_path / "a" / "img1.xml")
    assert (w, h) == (400, 200)
    assert boxes == [(100.0, 50.0, 300.0, 150.0)]  # the "car" object is ignored


def test_prepare_split_is_disjoint_and_labels_written(tmp_path):
    for i in range(10):
        _write(tmp_path / "i", tmp_path / "a", f"Plate ({i})")
    stats = prepare([(tmp_path / "i", tmp_path / "a")], tmp_path / "out", val_fraction=0.2, seed=1)
    assert stats["used"] == 10 and stats["train"] == 8 and stats["val"] == 2
    train = {p.stem for p in (tmp_path / "out/images/train").iterdir()}
    val = {p.stem for p in (tmp_path / "out/images/val").iterdir()}
    assert train.isdisjoint(val)
    for stem in train:
        assert (tmp_path / "out/labels/train" / f"{stem}.txt").read_text().startswith("0 ")
    assert (tmp_path / "out/data.yaml").exists()


def test_prepare_skips_size_mismatch(tmp_path):
    for i in range(4):
        _write(tmp_path / "i", tmp_path / "a", f"ok{i}")
    _write(tmp_path / "i", tmp_path / "a", "rotated", img_w=400, img_h=200, xml_w=200, xml_h=400)  # swapped dims
    stats = prepare([(tmp_path / "i", tmp_path / "a")], tmp_path / "out")
    assert stats["used"] == 4 and stats["skipped"] == 1


def test_prepare_scales_boxes_when_xml_size_is_proportionally_different(tmp_path):
    _write(tmp_path / "i", tmp_path / "a", "a1", img_w=400, img_h=200, xml_w=200, xml_h=100, box=(50, 25, 150, 75))
    _write(tmp_path / "i", tmp_path / "a", "a2")
    prepare([(tmp_path / "i", tmp_path / "a")], tmp_path / "out", val_fraction=0.5)
    labels = [p.read_text() for p in (tmp_path / "out/labels").rglob("*.txt")]
    assert all(line.startswith("0 0.500000 0.500000 0.500000 0.500000") for line in labels)


def test_prepare_downscales_large_images(tmp_path):
    _write(tmp_path / "i", tmp_path / "a", "big1", img_w=2000, img_h=1000, box=(500, 250, 1500, 750))
    _write(tmp_path / "i", tmp_path / "a", "big2", img_w=2000, img_h=1000, box=(500, 250, 1500, 750))
    prepare([(tmp_path / "i", tmp_path / "a")], tmp_path / "out", max_side=500, val_fraction=0.5)
    img = cv2.imread(str(next((tmp_path / "out/images").rglob("*.jpg"))))
    assert max(img.shape[:2]) == 500


def test_prepare_needs_two_usable_images(tmp_path):
    _write(tmp_path / "i", tmp_path / "a", "only")
    with pytest.raises(SystemExit):
        prepare([(tmp_path / "i", tmp_path / "a")], tmp_path / "out")


# ---------------------------------------------------------------------------
# YOLO-format sources, hold-out validation, caps
# ---------------------------------------------------------------------------
from scripts.prepare_plate_dataset import read_yolo_labels  # noqa: E402


def _make_yolo(root: Path, n: int, lines: str = "0 0.5 0.5 0.4 0.4\n1 0.2 0.2 0.1 0.1\n"):
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(root / "images" / f"y{i}.jpg"), np.zeros((100, 200, 3), dtype=np.uint8))
        (root / "labels" / f"y{i}.txt").write_text(lines)


def test_read_yolo_labels_filters_class_and_formats(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text(
        "0 0.5 0.5 0.4 0.4\n"          # plate
        "1 0.2 0.2 0.1 0.1\n"          # other class -> dropped
        "0 0.5 0.5 0.2 0.2 0.93\n"     # 6th field = confidence -> kept
        "0 0.1 0.1 0.2 0.1 0.3 0.3 0.4 0.4\n"  # polygon -> dropped
        "garbage line\n"
    )
    assert len(read_yolo_labels(label, {0})) == 2
    assert len(read_yolo_labels(label, {1})) == 1


def test_read_yolo_labels_clips_to_image(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("0 0.95 0.5 0.3 0.4\n")  # sticks out of the right edge
    assert read_yolo_labels(label, {0}) == ["0 0.900000 0.500000 0.200000 0.400000"]


def test_prepare_holdout_trains_on_yolo_validates_on_voc(tmp_path):
    _make_yolo(tmp_path / "big", 12)
    _write(tmp_path / "vi", tmp_path / "va", "hand1")
    _write(tmp_path / "vi", tmp_path / "va", "hand2")
    stats = prepare(
        [(tmp_path / "vi", tmp_path / "va")], tmp_path / "out",
        yolo_pairs=[(tmp_path / "big" / "images", tmp_path / "big" / "labels")], holdout_voc=True,
    )
    assert (stats["train"], stats["val"]) == (12, 2)
    assert all(p.name.startswith("yolo") for p in (tmp_path / "out/images/train").iterdir())
    assert all(p.name.startswith("voc") for p in (tmp_path / "out/images/val").iterdir())
    first = next((tmp_path / "out/labels/train").iterdir()).read_text().strip().splitlines()
    assert first == ["0 0.500000 0.500000 0.400000 0.400000"]  # class-1 line dropped


def test_prepare_max_train_images_caps_training_set(tmp_path):
    _make_yolo(tmp_path / "big", 30)
    _write(tmp_path / "vi", tmp_path / "va", "hand1")
    stats = prepare(
        [(tmp_path / "vi", tmp_path / "va")], tmp_path / "out",
        yolo_pairs=[(tmp_path / "big" / "images", tmp_path / "big" / "labels")],
        holdout_voc=True, max_train_images=10,
    )
    assert stats["train"] == 10 and stats["val"] == 1


def test_prepare_yolo_skips_images_without_plate_label(tmp_path):
    _make_yolo(tmp_path / "big", 5)
    (tmp_path / "big" / "labels" / "y0.txt").unlink()              # no label file
    (tmp_path / "big" / "labels" / "y1.txt").write_text("1 .5 .5 .1 .1\n")  # wrong class only
    stats = prepare([], tmp_path / "out", yolo_pairs=[(tmp_path / "big" / "images", tmp_path / "big" / "labels")])
    assert stats["used"] == 3 and stats["skipped"] == 2


def test_inspector_reports_yolo_class_ids(tmp_path):
    import subprocess
    import sys

    _make_yolo(tmp_path / "big", 4)
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "inspect_plate_dataset.py"), "--root", str(tmp_path / "big")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "YOLO label analysis" in result.stdout
    assert "class ids: {0: 4, 1: 4}" in result.stdout


# ---------------------------------------------------------------------------
# --yolo-exts, --max-box-area, inspector buckets, preview sheet
# ---------------------------------------------------------------------------
def _make_mixed_yolo(root: Path):
    """3 small-plate .jpg scenes + 4 tight-crop .png images (box covers ~90% of the image)."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(3):
        cv2.imwrite(str(root / "images" / f"scene{i}.jpg"), np.zeros((200, 300, 3), dtype=np.uint8))
        (root / "labels" / f"scene{i}.txt").write_text("0 0.5 0.6 0.2 0.1\n")
    for i in range(4):
        cv2.imwrite(str(root / "images" / f"crop{i}.png"), np.zeros((60, 200, 3), dtype=np.uint8))
        (root / "labels" / f"crop{i}.txt").write_text("0 0.5 0.5 0.95 0.9\n")


def test_yolo_exts_keeps_only_jpg(tmp_path):
    _make_mixed_yolo(tmp_path / "big")
    stats = prepare([], tmp_path / "out", yolo_pairs=[(tmp_path / "big/images", tmp_path / "big/labels")], yolo_exts=[".jpg"])
    assert stats["used"] == 3 and stats["skipped"] == 0


def test_max_box_area_drops_tight_crops(tmp_path):
    _make_mixed_yolo(tmp_path / "big")
    stats = prepare([], tmp_path / "out", yolo_pairs=[(tmp_path / "big/images", tmp_path / "big/labels")], max_box_area=0.5)
    assert stats["used"] == 3 and stats["skipped"] == 4


def test_inspector_shows_buckets_and_split_by_image_type(tmp_path):
    import subprocess
    import sys

    _make_mixed_yolo(tmp_path / "big")
    root = Path(__file__).resolve().parent.parent
    out = subprocess.run(
        [sys.executable, str(root / "scripts" / "inspect_plate_dataset.py"), "--root", str(tmp_path / "big")],
        capture_output=True, text=True,
    ).stdout
    assert "box size buckets" in out and "'>50%': 4" in out
    assert "by image type .png: 4 boxes" in out and "by image type .jpg: 3 boxes" in out


def test_preview_sheet_is_written(tmp_path):
    import subprocess
    import sys

    _make_mixed_yolo(tmp_path / "big")
    root = Path(__file__).resolve().parent.parent
    sheet = tmp_path / "sheet.jpg"
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "preview_yolo_labels.py"), "--images", str(tmp_path / "big/images"),
         "--labels", str(tmp_path / "big/labels"), "--ext", ".png", "--n", "4", "--out", str(sheet)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    img = cv2.imread(str(sheet))
    assert img.shape[1] == 480 * 4 and img.shape[0] == 360  # 1 row x 4 columns