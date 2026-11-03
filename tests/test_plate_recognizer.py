"""Tests for src/plate_recognizer.py.

No YOLO/EasyOCR models are loaded: the detector and OCR reader are small stubs injected through
the constructor, so these run fast in CI. They cover the static helpers and the accept/reject
decision (format validation + confidence), which is where ANPR output becomes a logged plate.
"""

import numpy as np
import pytest

from src.config import PLATE_UNCLEAR_LABEL
from src.plate_recognizer import PlateRecognizer


# ---------------------------------------------------------------- static helpers
def test_normalize_strips_whitespace_and_symbols():
    assert PlateRecognizer._normalize_plate_text("MH 12 AB 1234") == "MH12AB1234"
    assert PlateRecognizer._normalize_plate_text("mh-12-ab-1234") == "MH12AB1234"


def test_normalize_uppercases():
    assert PlateRecognizer._normalize_plate_text("ka01mf5678") == "KA01MF5678"


def test_parse_state_district_valid_plate():
    state, district = PlateRecognizer._parse_state_district("MH12AB1234")
    assert state == "MH"
    assert district == "12"


def test_parse_state_district_invalid_state_code():
    state, district = PlateRecognizer._parse_state_district("ZZ12AB1234")
    assert state is None
    assert district is None


def test_parse_state_district_single_digit_district():
    state, district = PlateRecognizer._parse_state_district("KA1MF5678")
    assert state == "KA"
    assert district == "1"


# ---------------------------------------------------------------- stubs
def _box(x, y, w=40, h=20):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


class StubReader:
    """EasyOCR stand-in: returns canned reads and records what it was called with."""

    def __init__(self, reads=None, error=None):
        self.reads = reads or []
        self.error = error
        self.calls = []

    def readtext(self, image, **kwargs):
        self.calls.append((image, kwargs))
        if self.error:
            raise self.error
        return self.reads


class _Tensor:
    def __init__(self, values):
        self.values = np.array(values)

    def argmax(self):
        return _Tensor(int(self.values.argmax()))

    def item(self):
        return self.values.item()

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def __getitem__(self, index):
        return _Tensor(self.values[index])

    def __len__(self):
        return len(self.values)


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy, self.conf = _Tensor(xyxy), _Tensor(conf)

    def __len__(self):
        return len(self.conf)


class StubDetector:
    """YOLO stand-in returning fixed plate boxes (or none)."""

    def __init__(self, xyxy=None, conf=None):
        self.boxes = _Boxes(xyxy, conf) if xyxy else None
        self.conf_used = None

    def predict(self, source, conf, verbose=False):
        self.conf_used = conf

        class _Result:
            pass

        result = _Result()
        result.boxes = self.boxes
        return [result]


def _recognizer(reads=None, threshold=0.3, error=None, detector=None):
    reader = StubReader(reads, error)
    return PlateRecognizer(confidence_threshold=threshold, detector=detector or StubDetector(), reader=reader), reader


def _crop(width=120, height=40):
    return np.full((height, width, 3), 200, dtype=np.uint8)


TWO_LINE_JH = [(_box(5, 60), "2117", 0.9), (_box(0, 5, 70), "JHO5AW", 0.8)]  # bottom line returned first, O for 0


# ---------------------------------------------------------------- read_plate_crop
def test_two_line_plate_is_reordered_corrected_and_accepted():
    recognizer, _ = _recognizer(TWO_LINE_JH)
    result = recognizer.read_plate_crop(_crop())
    assert result.plate_number == "JH05AW2117"
    assert result.is_valid_format
    assert (result.state_code, result.district_code) == ("JH", "05")
    assert result.raw_text == "JHO5AW2117"  # uncorrected reading-order text


def test_crop_is_enlarged_and_restricted_to_plate_characters():
    recognizer, reader = _recognizer(TWO_LINE_JH)
    recognizer.read_plate_crop(_crop(width=100))
    image, kwargs = reader.calls[0]
    assert image.ndim == 2 and image.shape[1] >= 400  # grayscale, enlarged
    assert kwargs["allowlist"] == "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def test_extra_marks_around_plate_are_ignored():
    recognizer, _ = _recognizer([(_box(0, 5), "IND", 0.9), (_box(60, 5), "KL07BX7197", 0.8)])
    assert recognizer.read_plate_crop(_crop()).plate_number == "KL07BX7197"


def test_delhi_plate_with_one_digit_district_is_accepted():
    recognizer, _ = _recognizer([(_box(0, 5, 100), "DL8CAZ4521", 0.8)])
    result = recognizer.read_plate_crop(_crop())
    assert result.plate_number == "DL8CAZ4521"
    assert result.district_code == "8"


def test_invalid_text_is_never_returned_as_a_plate_even_with_high_confidence():
    """Old behaviour logged garbage like BR331IND4980 as a plate when confidence was high."""
    recognizer, _ = _recognizer([(_box(0, 5, 100), "HEROBIKE", 0.99)])
    result = recognizer.read_plate_crop(_crop())
    assert result.plate_number == PLATE_UNCLEAR_LABEL
    assert not result.is_valid_format
    assert result.raw_text == "HEROBIKE"
    assert result.confidence == pytest.approx(0.99)


def test_valid_plate_below_confidence_threshold_is_unclear():
    recognizer, _ = _recognizer([(_box(0, 5, 100), "MH12AB1234", 0.2)], threshold=0.3)
    result = recognizer.read_plate_crop(_crop())
    assert result.plate_number == PLATE_UNCLEAR_LABEL
    assert result.is_valid_format  # format was fine, only the confidence failed


def test_moderate_confidence_valid_plate_is_accepted_at_default_threshold():
    """EasyOCR often scores right plates at 0.3-0.6; the old 0.7 cut-off threw them away."""
    from src.config import PLATE_CONFIDENCE_THRESHOLD

    recognizer, _ = _recognizer([(_box(0, 5, 100), "MH12AB1234", 0.45)], threshold=PLATE_CONFIDENCE_THRESHOLD)
    assert recognizer.read_plate_crop(_crop()).plate_number == "MH12AB1234"


def test_nothing_read_is_unclear_with_zero_confidence():
    recognizer, _ = _recognizer([])
    result = recognizer.read_plate_crop(_crop())
    assert result.plate_number == PLATE_UNCLEAR_LABEL and result.confidence == 0.0
    assert result.raw_text == ""


def test_ocr_exception_is_contained():
    recognizer, _ = _recognizer(error=RuntimeError("cuda out of memory"))
    result = recognizer.read_plate_crop(_crop())
    assert result.plate_number == PLATE_UNCLEAR_LABEL and not result.is_valid_format


def test_empty_crop_is_unclear():
    recognizer, reader = _recognizer(TWO_LINE_JH)
    result = recognizer.read_plate_crop(np.zeros((0, 0, 3), dtype=np.uint8))
    assert result.plate_number == PLATE_UNCLEAR_LABEL
    assert reader.calls == []


# ---------------------------------------------------------------- detect / read_plate
def test_read_plate_returns_none_when_no_plate_box_found():
    recognizer, reader = _recognizer(TWO_LINE_JH, detector=StubDetector())
    assert recognizer.read_plate(_crop(200, 100)) is None
    assert reader.calls == []


def test_read_plate_crops_the_highest_confidence_box_and_reads_it():
    detector = StubDetector(xyxy=[[0, 0, 50, 20], [10, 30, 150, 90]], conf=[0.4, 0.9])
    recognizer, reader = _recognizer(TWO_LINE_JH, detector=detector)
    result = recognizer.read_plate(_crop(200, 100))
    assert result.plate_number == "JH05AW2117"
    assert detector.conf_used == recognizer.detection_conf
    # the 140x60 box was passed to OCR, enlarged to the minimum width
    assert reader.calls[0][0].shape[1] >= 400


def test_degenerate_box_is_treated_as_no_plate():
    detector = StubDetector(xyxy=[[50, 50, 50, 50]], conf=[0.9])
    recognizer, _ = _recognizer(TWO_LINE_JH, detector=detector)
    assert recognizer.read_plate(_crop(200, 100)) is None
