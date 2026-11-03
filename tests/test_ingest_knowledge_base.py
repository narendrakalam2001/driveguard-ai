"""Tests for scripts/ingest_knowledge_base.py — knowledge base parsing into Documents."""

from scripts.ingest_knowledge_base import build_documents, document_id


def test_build_documents_covers_all_source_types():
    docs = build_documents()
    doc_types = {d.metadata["doc_type"] for d in docs}
    assert {"act_section", "fine_schedule", "speed_zones", "faq"}.issubset(doc_types)


def test_build_documents_nonempty():
    docs = build_documents()
    assert len(docs) > 15  # sanity floor given 8 sections + 7 fines + zones + faq chunks


def test_act_section_documents_reference_real_sections():
    docs = build_documents()
    act_docs = [d for d in docs if d.metadata["doc_type"] == "act_section"]
    section_numbers = {d.metadata["section"] for d in act_docs}
    assert "Section 183" in section_numbers  # speeding — the core violation type this project handles


def test_fine_schedule_documents_have_currency():
    docs = build_documents()
    fine_docs = [d for d in docs if d.metadata["doc_type"] == "fine_schedule"]
    assert all("INR" in d.page_content for d in fine_docs)


def test_document_ids_are_stable_and_unique_so_reingesting_does_not_duplicate():
    first = [document_id(d) for d in build_documents()]
    second = [document_id(d) for d in build_documents()]
    assert first == second            # same input -> same ids -> Chroma updates in place
    assert len(set(first)) == len(first)  # no two chunks collide
