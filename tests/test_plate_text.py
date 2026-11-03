"""Tests for src/plate_text.py - reading order and plate-pattern correction."""

import numpy as np

from src.plate_text import (
    OCR_ALLOWLIST,
    correct_plate_text,
    is_valid_plate,
    normalize,
    order_reads,
    prepare_crop,
    read_plate_text,
)


def _box(x, y, w=40, h=20):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def test_normalize():
    assert normalize("mh 12.ab-1234") == "MH12AB1234"


def test_order_reads_two_line_plate_top_line_first():
    reads = [(_box(5, 60), "2117", 0.9), (_box(0, 5, 70), "JH05AW", 0.8)]  # EasyOCR returned bottom line first
    assert [t for t, _ in order_reads(reads)] == ["JH05AW", "2117"]


def test_order_reads_single_line_left_to_right():
    reads = [(_box(100, 10), "1234", 0.9), (_box(0, 12), "MH12", 0.9), (_box(50, 9), "AB", 0.9)]
    assert "".join(t for t, _ in order_reads(reads)) == "MH12AB1234"


def test_order_reads_empty():
    assert order_reads([]) == []


def test_valid_plate_is_unchanged():
    assert correct_plate_text("MH 12 AB 1234") == "MH12AB1234"
    assert correct_plate_text("TN-18-D-1866") == "TN18D1866"


def test_look_alike_characters_are_fixed_in_digit_and_letter_positions():
    assert correct_plate_text("JHO5AW2117") == "JH05AW2117"   # O->0 in the district digits
    assert correct_plate_text("JHOSAW2117", max_cost=2) == "JH05AW2117"  # two swaps need max_cost=2
    assert correct_plate_text("JHOSAW2117") == "JHOSAW2117"           # default refuses two swaps
    assert correct_plate_text("MH12A81234") == "MH12AB1234"   # 8->B in the series letters


def test_extra_text_around_the_plate_is_ignored():
    assert correct_plate_text("INDKL07BX7197") == "KL07BX7197"
    assert correct_plate_text("HEROKA38M1674SOMETHING") == "KA38M1674"


def test_reads_with_no_plate_pattern_are_returned_unchanged():
    assert correct_plate_text("2117JHO5AW") == "2117JHO5AW"       # wrong order: nothing to fit
    assert correct_plate_text("XXXXXXXXXX") == "XXXXXXXXXX"
    assert correct_plate_text("") == ""


def test_invalid_state_code_is_not_accepted():
    assert correct_plate_text("ZZ12AB1234") == "ZZ12AB1234"


def test_more_than_max_cost_changes_are_refused():
    assert correct_plate_text("MHOSAB12S4", max_cost=1) == "MHOSAB12S4"
    assert correct_plate_text("MH12AB1234", max_cost=0) == "MH12AB1234"


def test_delhi_one_digit_district_layout_is_fitted():
    assert correct_plate_text("DL8CAZ4521") == "DL8CAZ4521"
    assert correct_plate_text("INDDL8CAZ4521") == "DL8CAZ4521"
    assert correct_plate_text("DL8SBT6438") == "DL8SBT6438"


def test_delhi_layout_is_not_used_for_other_states():
    """Otherwise JHO5AW2117 could be mis-read as JH0SAW2117 (O->0 in the district, S as a series letter)."""
    assert correct_plate_text("JHOSAW2117") == "JHOSAW2117"
    assert correct_plate_text("JHO5AW2117") == "JH05AW2117"


def test_is_valid_plate():
    assert is_valid_plate("MH12AB1234") and is_valid_plate("mh 12-ab 1234")
    assert is_valid_plate("DL8CAZ4521") and is_valid_plate("KA1MF5678")
    assert not is_valid_plate("ZZ12AB1234")       # no such state code
    assert not is_valid_plate("MH12AB123")        # number must have 4 digits
    assert not is_valid_plate("BR331IND4980")     # OCR garbage
    assert not is_valid_plate("")


def test_prepare_crop_makes_small_crops_wide_grayscale_and_keeps_big_ones():
    small = np.full((20, 100, 3), 120, dtype=np.uint8)
    out = prepare_crop(small, 400)
    assert out.ndim == 2 and out.shape[1] == 400 and out.shape[0] == 80
    big = np.full((50, 500, 3), 120, dtype=np.uint8)
    assert prepare_crop(big, 400).shape == (50, 500)
    gray = np.full((20, 100), 120, dtype=np.uint8)
    assert prepare_crop(gray, 400).shape[1] == 400


class _Reader:
    def __init__(self, reads):
        self.reads, self.kwargs = reads, None

    def readtext(self, image, **kwargs):
        self.kwargs = kwargs
        return self.reads


def test_read_plate_text_runs_the_full_flow():
    reads = [(_box(5, 60), "2117", 0.9), (_box(0, 5, 70), "JHO5AW", 0.6)]
    reader = _Reader(reads)
    text, confidence, raw = read_plate_text(reader, np.full((40, 120, 3), 200, dtype=np.uint8))
    assert (text, raw) == ("JH05AW2117", "JHO5AW2117")
    assert reader.kwargs == {"allowlist": OCR_ALLOWLIST}
    assert abs(confidence - (4 * 0.9 + 6 * 0.6) / 10) < 1e-9  # weighted by characters read


def test_read_plate_text_stray_short_fragment_barely_moves_confidence():
    reads = [(_box(0, 5, 100), "MH12AB1234", 0.5), (_box(0, 60), "IN", 0.99)]
    _, confidence, _ = read_plate_text(_Reader(reads), np.full((40, 120, 3), 200, dtype=np.uint8))
    assert 0.5 < confidence < 0.6


def test_read_plate_text_nothing_read():
    assert read_plate_text(_Reader([]), np.full((40, 120, 3), 200, dtype=np.uint8)) == ("", 0.0, "")
    assert read_plate_text(_Reader([]), np.zeros((0, 0, 3), dtype=np.uint8)) == ("", 0.0, "")


# ---------------------------------------------------------------------------
# Plate regions: the product is Indian, UK/ANY exist to validate OCR on non-Indian footage
# ---------------------------------------------------------------------------
def test_uk_plates_are_rejected_by_the_default_indian_rules_but_accepted_in_uk_mode():
    from src.plate_text import is_valid_plate

    for plate in ("GX15OGJ", "NA54KGJ", "LN15ZZC", "GJ06EPD"):   # plates visible in the UK test footage
        assert not is_valid_plate(plate)          # why 586 plate boxes produced 0 reads
        assert is_valid_plate(plate, "UK")
    assert not is_valid_plate("MH12AB1234", "UK") and is_valid_plate("MH12AB1234")


def test_any_region_needs_letters_and_digits_in_a_sane_length():
    from src.plate_text import is_valid_plate

    assert is_valid_plate("AB12CD", "ANY") and is_valid_plate("1234AB", "ANY")
    assert not is_valid_plate("ABCDEF", "ANY") and not is_valid_plate("123456", "ANY") and not is_valid_plate("A1", "ANY")


def test_uk_correction_fixes_lookalikes_by_position_and_ignores_extra_text():
    from src.plate_text import correct_plate_text

    assert correct_plate_text("GX15OGJ", region="UK") == "GX15OGJ"
    assert correct_plate_text("GX150GJ", region="UK") == "GX15OGJ"      # a zero where a letter belongs
    assert correct_plate_text("GXI5OGJ", region="UK") == "GX15OGJ"      # an I where a digit belongs
    assert correct_plate_text("UK GX15 OGJ GB", region="UK") == "GX15OGJ"
    assert correct_plate_text("12", region="UK") == "12"                 # nothing fits: unchanged


def test_display_formats_uk_plates_with_a_space():
    from src.plate_text import format_plate_for_display

    assert format_plate_for_display("GX15OGJ") == "GX15 OGJ"
    assert format_plate_for_display("MH12AB1234") == "MH 12 AB 1234"


def test_recognizer_region_decides_whether_the_same_uk_plate_is_read(monkeypatch):
    import numpy as np

    from src.plate_recognizer import PlateRecognizer

    class Reader:
        def readtext(self, image, allowlist=None):
            return [([[0, 0], [100, 0], [100, 30], [0, 30]], "GX15 0GJ", 0.8)]

    crop = np.full((30, 100, 3), 120, dtype=np.uint8)
    indian = PlateRecognizer(detector=object(), reader=Reader(), region="IN").read_plate_crop(crop)
    uk = PlateRecognizer(detector=object(), reader=Reader(), region="UK").read_plate_crop(crop)
    assert indian.plate_number == "PLATE_UNCLEAR" and indian.raw_text == "GX150GJ"   # OCR was right, the format check said no
    assert uk.plate_number == "GX15OGJ" and uk.state_code is None and uk.is_valid_format
