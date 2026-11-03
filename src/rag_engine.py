"""
rag_engine.py
==============
Query-side interface to the ChromaDB knowledge base built by
``scripts/ingest_knowledge_base.py``. Given a violation record, returns
the most relevant MV Act sections, fine schedule entries, and FAQ
context to ground the LLM's generated notice.

Two kinds of lookup, on purpose:
  * :class:`StructuredKnowledge` - exact lookups straight from the JSON files (which section
    applies to a violation, what the fine is). A fine amount or section number on an official
    notice must never depend on which chunk a vector search happened to rank first.
  * :class:`RAGEngine` - semantic search over the same files plus the FAQ, for explanatory
    context around those exact facts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.config import CHROMADB_STORE_DIR, KNOWLEDGE_BASE_DIR, RAG_COLLECTION_NAME, RAG_TOP_K, get_logger

logger = get_logger(__name__)

_SECTION_NUMBER_RE = re.compile(r"\bSection\s+(\d+[A-Z]?)", re.IGNORECASE)


def _inr(amount: int) -> str:
    """Format an amount the Indian way, e.g. 1000 -> '₹1,000', 150000 -> '₹1,50,000'."""
    digits = str(int(amount))
    if len(digits) <= 3:
        return f"₹{digits}"
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return "₹" + ",".join(groups + [tail])


@dataclass
class FineInfo:
    """Fine and legal section for one violation type, read exactly from ``fine_schedule.json``.

    Attributes:
        violation_type: Key such as "SPEEDING".
        section: Section label, e.g. "Section 183".
        first_range: Formatted first-offence range, e.g. "₹1,000 - ₹2,000".
        subsequent_range: Formatted repeat-offence range.
        notes: The schedule's own caveats for this entry.
        disclaimer: The file-level "indicative amounts only" disclaimer.
    """

    violation_type: str
    section: str
    first_range: str
    subsequent_range: str
    notes: str
    disclaimer: str

    def summary(self) -> str:
        """One line in the shape used on the notice."""
        return f"{self.first_range} (first offense) | {self.subsequent_range} (subsequent)"

    def applicable(self, repeat_offender: bool) -> str:
        """The range that applies to this offender."""
        return self.subsequent_range if repeat_offender else self.first_range


class StructuredKnowledge:
    """Exact (non-semantic) access to the MV Act sections and the fine schedule."""

    def __init__(self, kb_dir: Path = KNOWLEDGE_BASE_DIR) -> None:
        """Load both JSON files once.

        Args:
            kb_dir: Folder holding ``mv_act_sections.json`` and ``fine_schedule.json``.

        Raises:
            KnowledgeBaseNotFoundError: If either file is missing.
        """
        sections_path, fines_path = kb_dir / "mv_act_sections.json", kb_dir / "fine_schedule.json"
        for path in (sections_path, fines_path):
            if not path.exists():
                raise KnowledgeBaseNotFoundError(f"Knowledge base file not found: {path}")
        self._sections = json.loads(sections_path.read_text(encoding="utf-8"))
        schedule = json.loads(fines_path.read_text(encoding="utf-8"))
        self._disclaimer = schedule.get("_disclaimer", "")
        self._fines = {k: v for k, v in schedule.items() if not k.startswith("_")}

    def get_fine(self, violation_type: str) -> Optional[FineInfo]:
        """Return the fine entry for a violation type, or None if the schedule has no such key."""
        entry = self._fines.get(violation_type.upper().replace(" ", "_"))
        if entry is None:
            return None
        return FineInfo(
            violation_type=violation_type.upper(),
            section=entry["section"],
            first_range=self._range(entry["first_offense_min"], entry["first_offense_max"]),
            subsequent_range=self._range(entry["subsequent_offense_min"], entry["subsequent_offense_max"]),
            notes=entry.get("notes", ""),
            disclaimer=self._disclaimer,
        )

    @staticmethod
    def _range(low: int, high: int) -> str:
        return _inr(low) if low == high else f"{_inr(low)} - {_inr(high)}"

    def get_sections(self, violation_type: str, severity: str) -> List[dict]:
        """MV Act sections that apply to a violation, from each section's own ``applies_to`` list.

        SEVERE speeding also returns the dangerous-driving section (184).

        Args:
            violation_type: e.g. "SPEEDING".
            severity: WARNING / VIOLATION / SEVERE.

        Returns:
            Matching section dicts (``section``, ``title``, ``summary`` ...), base section first.
        """
        wanted = [violation_type.upper()]
        if violation_type.upper() == "SPEEDING" and severity == "SEVERE":
            wanted.append("SEVERE_SPEEDING")
        found: List[dict] = []
        for tag in wanted:
            for section in self._sections:
                if tag in section.get("applies_to", []) and section not in found:
                    found.append(section)
        return found

    @staticmethod
    def section_numbers(text: str) -> set:
        """All 'Section NNN' numbers mentioned in a piece of text (e.g. {'183', '184'})."""
        return {m.upper() for m in _SECTION_NUMBER_RE.findall(text)}


class KnowledgeBaseNotFoundError(RuntimeError):
    """Raised when the ChromaDB store hasn't been ingested yet."""


@dataclass
class RetrievedChunk:
    """One piece of retrieved knowledge-base context.

    Attributes:
        content: The chunk's text.
        source: Source file name (e.g. "mv_act_sections.json").
        doc_type: One of "act_section", "fine_schedule", "speed_zones", "faq".
        score: Similarity score (lower = more similar, for Chroma's default L2/cosine distance).
        metadata: Full metadata dict for downstream use (e.g. citing the exact section).
    """

    content: str
    source: str
    doc_type: str
    score: float
    metadata: dict


class RAGEngine:
    """Retrieval-augmented context provider over the MV Act knowledge base."""

    def __init__(
        self,
        collection_name: str = RAG_COLLECTION_NAME,
        persist_directory=CHROMADB_STORE_DIR,
        top_k: int = RAG_TOP_K,
    ) -> None:
        """Connect to the existing (already-ingested) Chroma collection.

        Args:
            collection_name: Name of the Chroma collection to query.
            persist_directory: Directory where Chroma's on-disk store lives.
            top_k: Default number of chunks to retrieve per query.

        Raises:
            KnowledgeBaseNotFoundError: If the persist directory doesn't
                exist yet (i.e. ingestion hasn't been run).
        """
        from langchain_chroma import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings

        if not persist_directory.exists():
            raise KnowledgeBaseNotFoundError(
                f"No ChromaDB store found at {persist_directory}. "
                "Run scripts/ingest_knowledge_base.py first."
            )

        self.top_k = top_k
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self._vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_directory),
        )
        logger.info(f"RAGEngine connected to collection='{collection_name}' at {persist_directory}")

    def retrieve(self, query: str, top_k: Optional[int] = None, doc_type: Optional[str] = None) -> List[RetrievedChunk]:
        """Retrieve the most relevant knowledge-base chunks for a query.

        Args:
            query: Natural-language or structured query string.
            top_k: Override the default number of results.
            doc_type: Optionally restrict results to one doc_type
                ("act_section", "fine_schedule", "speed_zones", "faq").

        Returns:
            List of :class:`RetrievedChunk`, best match first.
        """
        k = top_k or self.top_k
        filter_dict = {"doc_type": doc_type} if doc_type else None

        results = self._vectorstore.similarity_search_with_score(query, k=k, filter=filter_dict)
        chunks = [
            RetrievedChunk(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                doc_type=doc.metadata.get("doc_type", "unknown"),
                score=float(score),
                metadata=doc.metadata,
            )
            for doc, score in results
        ]
        return chunks

    def get_context_for_violation(self, violation_type: str, severity: str) -> List[RetrievedChunk]:
        """Convenience method: pull the MV Act section + fine schedule for a violation.

        Combines a targeted act-section lookup with a fine-schedule lookup
        so :mod:`report_generator` gets exactly what it needs in one call.

        Args:
            violation_type: e.g. "SPEEDING", "OVERLOADING".
            severity: One of WARNING / VIOLATION / SEVERE — folded into the
                query so severe cases also surface the dangerous-driving
                section (184) alongside the base speeding section (183).

        Returns:
            Combined, de-duplicated list of relevant chunks.
        """
        query_parts = [violation_type.replace("_", " ")]
        if severity == "SEVERE":
            query_parts.append("dangerous driving")
        query = " ".join(query_parts)

        act_chunks = self.retrieve(query, top_k=2, doc_type="act_section")
        fine_chunks = self.retrieve(violation_type.replace("_", " "), top_k=1, doc_type="fine_schedule")

        combined = act_chunks + fine_chunks
        seen = set()
        deduped = []
        for chunk in combined:
            key = chunk.content[:50]
            if key not in seen:
                seen.add(key)
                deduped.append(chunk)
        return deduped

    def format_context_block(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks into a single context block for the LLM prompt.

        Args:
            chunks: Chunks from :meth:`retrieve` or :meth:`get_context_for_violation`.

        Returns:
            A newline-joined, source-attributed context string.
        """
        lines = []
        for chunk in chunks:
            lines.append(f"[{chunk.source}] {chunk.content}")
        return "\n".join(lines)
