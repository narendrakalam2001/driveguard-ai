"""Tests for src/report_generator.py with a stub Gemini client (no network, no API key)."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.report_generator import FALLBACK_MODEL_NAME, ReportGenerator
from src.violation_classifier import ViolationRecord


def _record(severity="SEVERE", plate="MH12AB1234", speed=87.0, limit=60):
    excess = speed - limit
    return ViolationRecord(
        violation_id="v-1", track_id=7, plate_number=plate, vehicle_type="Truck",
        detected_speed_kmph=speed, speed_limit_kmph=limit, excess_speed_kmph=excess,
        excess_percent=round(excess / limit * 100, 1), severity=severity,
        location="NH-48, Pune-Mumbai Highway", camera_id="CAM-1",
        timestamp=datetime(2024, 1, 15, 14, 32, 45, tzinfo=timezone.utc),
    )


GOOD_NOTICE = (
    "TRAFFIC VIOLATION NOTICE\n\nThis is to inform that vehicle MH 12 AB 1234 (Truck) was detected travelling "
    "at 87 km/h in a 60 km/h zone on NH-48, Pune-Mumbai Highway at 14:32:45.\n\nThis constitutes a violation of "
    "Section 183 of the Motor Vehicles Act, 1988.\n\nApplicable Fine: ₹1,000 - ₹2,000 (first offense). Amounts are "
    "indicative.\n\nReport to the nearest RTO within 30 days."
)


class StubClient:
    """Mimics ``google.genai.Client`` just enough: ``client.models.generate_content(...)``."""

    def __init__(self, text=GOOD_NOTICE, error=None):
        self.text, self.error, self.calls = text, error, []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.text)


class NoRag:
    def retrieve(self, *a, **k):
        raise AssertionError("semantic search must not be needed")


def _generator(client=None, **kw):
    gen = ReportGenerator(rag_engine=NoRag(), api_key="test-key", model_name="gemini-test", client=client, **kw)
    gen._rag_unavailable = True  # exact knowledge only; no Chroma
    return gen


def test_good_llm_notice_is_accepted_and_grounded_in_the_exact_fine():
    client = StubClient()
    notice = _generator(client).generate(_record(), vehicle_id="TRK_007")
    assert notice.model_used == "gemini-test"
    assert notice.notice_text == GOOD_NOTICE
    assert notice.cited_context[0].startswith("Applicable Fine (Section 183): ₹1,000 - ₹2,000 [first offense]")
    call = client.calls[0]
    assert call["model"] == "gemini-test"
    assert "Applicable Fine (Section 183): ₹1,000 - ₹2,000" in call["contents"]
    assert "official notice-drafting assistant" in call["config"].system_instruction


def test_severe_speeding_context_includes_dangerous_driving_but_warning_does_not():
    severe = _generator(StubClient()).generate(_record("SEVERE"), "TRK_007")
    assert any(line.startswith("Section 184") for line in severe.cited_context) or severe.model_used == FALLBACK_MODEL_NAME
    ctx, _ = _generator().build_context("SPEEDING", "SEVERE", False)
    assert any(line.startswith("Section 184") for line in ctx)
    ctx, _ = _generator().build_context("SPEEDING", "WARNING", False)
    assert not any("Section 184" in line for line in ctx)


def test_repeat_offender_gets_the_subsequent_offence_range():
    ctx, fine = _generator().build_context("SPEEDING", "VIOLATION", True)
    assert "Applicable Fine (Section 183): ₹2,000 - ₹4,000 [subsequent offense]" in ctx[0]
    assert fine.first_range == "₹1,000 - ₹2,000"


@pytest.mark.parametrize(
    "bad_text, reason",
    [
        (GOOD_NOTICE.replace("Section 183", "Section 999"), "sections"),
        (GOOD_NOTICE.replace("₹1,000 - ₹2,000", "₹50,000"), "fine amounts"),
        (GOOD_NOTICE.replace("MH 12 AB 1234", "an unnamed vehicle"), "plate"),
        (GOOD_NOTICE.replace("87 km/h", "an excessive speed"), "speed"),
        ("Too short.", "short"),
    ],
)
def test_llm_notice_that_breaks_the_rules_is_replaced_by_the_template(bad_text, reason):
    notice = _generator(StubClient(bad_text)).generate(_record(), "TRK_007")
    assert notice.model_used == FALLBACK_MODEL_NAME
    assert "MH12AB1234" in notice.notice_text and "Section 183" in notice.notice_text
    assert "₹1,000 - ₹2,000" in notice.notice_text
    assert notice.cited_context  # still audit-able


def test_unreadable_plate_notice_does_not_need_a_plate():
    text = GOOD_NOTICE.replace("MH 12 AB 1234", "a vehicle whose plate could not be read")
    notice = _generator(StubClient(text)).generate(_record(plate=None), "TRK_007")
    assert notice.model_used == "gemini-test"


def test_gemini_errors_propagate_so_the_api_can_fall_back():
    with pytest.raises(RuntimeError):
        _generator(StubClient(error=RuntimeError("404 model not found"))).generate(_record(), "TRK_007")


def test_missing_api_key_is_a_clear_error():
    gen = ReportGenerator(rag_engine=NoRag(), api_key="", model_name="gemini-test")
    gen._rag_unavailable = True
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        gen.generate(_record(), "TRK_007")


def test_fallback_notice_has_plate_section_fine_and_no_llm_or_semantic_search():
    client = StubClient()
    gen = ReportGenerator(rag_engine=NoRag(), api_key="k", client=client)  # NoRag raises if searched
    notice = gen.generate_fallback_notice(_record(), "TRK_007", is_repeat_offender=True)
    assert client.calls == []
    assert notice.model_used == FALLBACK_MODEL_NAME
    text = notice.notice_text
    assert "MH12AB1234" in text and "87 km/h" in text and "Section 183" in text
    assert "₹1,000 - ₹2,000 (first offense) | ₹2,000 - ₹4,000 (subsequent)" in text
    assert "Repeat offender: Yes" in text and "within 30 days" in text


def test_fallback_notice_for_unreadable_plate():
    notice = _generator().generate_fallback_notice(_record(plate=None), "TRK_007")
    assert "UNREADABLE" in notice.notice_text


def test_fallback_still_works_if_knowledge_files_are_missing():
    class Broken:
        def __getattr__(self, name):
            raise RuntimeError("knowledge base gone")

    gen = ReportGenerator(rag_engine=NoRag(), api_key="k", knowledge=Broken())
    notice = gen.generate_fallback_notice(_record(), "TRK_007")
    assert "Section 183" in notice.notice_text and "Applicable Fine" not in notice.notice_text


def test_semantic_search_failure_is_swallowed_and_not_retried():
    class FailingRag:
        calls = 0

        def retrieve(self, *a, **k):
            FailingRag.calls += 1
            raise RuntimeError("chroma store not ingested")

    gen = ReportGenerator(rag_engine=FailingRag(), api_key="k", model_name="m")
    first, fine = gen.build_context("SPEEDING", "VIOLATION", False)
    second, _ = gen.build_context("SPEEDING", "VIOLATION", False)
    assert fine is not None and first[0].startswith("Applicable Fine") and first == second
    assert FailingRag.calls == 1  # gave up after the first failure instead of retrying per notice


def test_semantic_chunks_are_appended_after_the_exact_facts():
    class Chunk:
        content = "FAQ: speeding challans can be contested within 30 days."

    class Rag:
        def retrieve(self, query, top_k, doc_type):
            assert doc_type == "faq"
            return [Chunk()]

    gen = ReportGenerator(rag_engine=Rag(), api_key="k")
    ctx, _ = gen.build_context("SPEEDING", "WARNING", False)
    assert ctx[0].startswith("Applicable Fine") and ctx[-1].startswith("FAQ:")
