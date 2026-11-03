"""Tests for src/rag_engine.py - exact fine/section lookups and the retrieval wrapper (no Chroma needed)."""

import pytest

from src.rag_engine import KnowledgeBaseNotFoundError, RAGEngine, StructuredKnowledge, _inr


@pytest.fixture(scope="module")
def knowledge():
    return StructuredKnowledge()


@pytest.mark.parametrize(
    "amount, expected",
    [(500, "₹500"), (1000, "₹1,000"), (20000, "₹20,000"), (150000, "₹1,50,000"), (10000000, "₹1,00,00,000")],
)
def test_inr_uses_indian_digit_grouping(amount, expected):
    assert _inr(amount) == expected


def test_speeding_fine_is_read_exactly_from_the_schedule(knowledge):
    fine = knowledge.get_fine("SPEEDING")
    assert fine.section == "Section 183"
    assert fine.first_range == "₹1,000 - ₹2,000"
    assert fine.subsequent_range == "₹2,000 - ₹4,000"
    assert fine.summary() == "₹1,000 - ₹2,000 (first offense) | ₹2,000 - ₹4,000 (subsequent)"
    assert "indicative" in fine.disclaimer.lower()


def test_fine_range_collapses_when_min_equals_max(knowledge):
    assert knowledge.get_fine("NO_HELMET").first_range == "₹1,000"


def test_applicable_range_depends_on_repeat_offender(knowledge):
    fine = knowledge.get_fine("SPEEDING")
    assert fine.applicable(False) == "₹1,000 - ₹2,000"
    assert fine.applicable(True) == "₹2,000 - ₹4,000"


def test_fine_lookup_ignores_case_and_spaces_and_unknown_is_none(knowledge):
    assert knowledge.get_fine("speeding").section == "Section 183"
    assert knowledge.get_fine("red light").section.startswith("Section 177")
    assert knowledge.get_fine("TELEPORTING") is None
    assert knowledge.get_fine("_disclaimer") is None


def test_sections_for_ordinary_speeding_is_only_183(knowledge):
    assert [s["section"] for s in knowledge.get_sections("SPEEDING", "WARNING")] == ["Section 183"]
    assert [s["section"] for s in knowledge.get_sections("SPEEDING", "VIOLATION")] == ["Section 183"]


def test_severe_speeding_also_returns_dangerous_driving_section(knowledge):
    assert [s["section"] for s in knowledge.get_sections("SPEEDING", "SEVERE")] == ["Section 183", "Section 184"]


def test_sections_for_unknown_violation_is_empty(knowledge):
    assert knowledge.get_sections("TELEPORTING", "SEVERE") == []


def test_section_numbers_extracts_numbers_only():
    text = "Under Section 183 and section 184, read with Section 194D; amounts in rupees."
    assert StructuredKnowledge.section_numbers(text) == {"183", "184", "194D"}


def test_missing_knowledge_files_raise_a_clear_error(tmp_path):
    with pytest.raises(KnowledgeBaseNotFoundError):
        StructuredKnowledge(tmp_path)


# ---------------------------------------------------------------- RAGEngine wrapper
class _Doc:
    def __init__(self, text, **meta):
        self.page_content, self.metadata = text, meta


class _FakeStore:
    def __init__(self):
        self.calls = []

    def similarity_search_with_score(self, query, k, filter=None):
        self.calls.append((query, k, filter))
        return [(_Doc("Answer text", source="traffic_faq.md", doc_type="faq"), 0.12)]


def _engine():
    engine = RAGEngine.__new__(RAGEngine)  # skip loading Chroma/embeddings
    engine.top_k, engine._vectorstore = 4, _FakeStore()
    return engine


def test_retrieve_passes_filter_and_k_and_wraps_results():
    engine = _engine()
    chunks = engine.retrieve("helmet", top_k=2, doc_type="faq")
    assert engine._vectorstore.calls == [("helmet", 2, {"doc_type": "faq"})]
    assert chunks[0].content == "Answer text" and chunks[0].doc_type == "faq" and chunks[0].score == pytest.approx(0.12)


def test_retrieve_without_doc_type_sends_no_filter():
    engine = _engine()
    engine.retrieve("speeding")
    assert engine._vectorstore.calls[0] == ("speeding", 4, None)


def test_format_context_block_attributes_each_chunk_to_its_source():
    engine = _engine()
    block = engine.format_context_block(engine.retrieve("x"))
    assert block == "[traffic_faq.md] Answer text"
