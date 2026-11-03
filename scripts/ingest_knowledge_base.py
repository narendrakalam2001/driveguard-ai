#!/usr/bin/env python3
"""
ingest_knowledge_base.py
==========================
Chunks and embeds the RAG knowledge base (MV Act sections, fine
schedule, speed limits, traffic FAQ) into a persistent ChromaDB
collection, consumed at query time by ``src/rag_engine.py``.

Usage:
    python scripts/ingest_knowledge_base.py
    python scripts/ingest_knowledge_base.py --rebuild   # wipe and re-embed everything
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import (
    CHROMADB_STORE_DIR,
    KNOWLEDGE_BASE_DIR,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_COLLECTION_NAME,
    get_logger,
)

logger = get_logger(__name__)


def _mv_act_sections_to_documents(path: Path) -> List[Document]:
    """Turn ``mv_act_sections.json`` into one Document per section.

    Each section is kept as a single chunk (they're already short and
    semantically atomic) rather than being run through the text
    splitter, so retrieval never returns a truncated section summary.

    Args:
        path: Path to mv_act_sections.json.

    Returns:
        List of LangChain Documents, one per Act section.
    """
    sections = json.loads(path.read_text())
    docs = []
    for section in sections:
        content = (
            f"{section['section']} — {section['title']}\n"
            f"{section['summary']}\n"
            f"Applies to: {', '.join(section['applies_to'])}"
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "mv_act_sections.json",
                    "section": section["section"],
                    "applies_to": ",".join(section["applies_to"]),
                    "doc_type": "act_section",
                },
            )
        )
    logger.info(f"Parsed {len(docs)} MV Act sections from {path}")
    return docs


def _fine_schedule_to_documents(path: Path) -> List[Document]:
    """Turn ``fine_schedule.json`` into one Document per violation type.

    Args:
        path: Path to fine_schedule.json.

    Returns:
        List of LangChain Documents, one per violation type entry.
    """
    schedule = json.loads(path.read_text())
    docs = []
    for violation_type, details in schedule.items():
        if violation_type.startswith("_"):
            continue
        content = (
            f"Fine schedule for {violation_type} ({details['section']}): "
            f"first offense {details['first_offense_min']}-{details['first_offense_max']} "
            f"{details['currency']}, subsequent offense "
            f"{details['subsequent_offense_min']}-{details['subsequent_offense_max']} "
            f"{details['currency']}. {details.get('notes', '')}"
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "fine_schedule.json",
                    "violation_type": violation_type,
                    "section": details["section"],
                    "doc_type": "fine_schedule",
                },
            )
        )
    logger.info(f"Parsed {len(docs)} fine schedule entries from {path}")
    return docs


def _speed_limits_to_documents(path: Path) -> List[Document]:
    """Turn ``speed_limits.json`` into a small set of Documents.

    Args:
        path: Path to speed_limits.json.

    Returns:
        List of LangChain Documents covering zone limits and state notes.
    """
    data = json.loads(path.read_text())
    docs = []

    zone_content = "Default zone speed limits: " + ", ".join(
        f"{zone} = {limit} kmph" for zone, limit in data["default_zone_limits_kmph"].items()
    )
    docs.append(
        Document(page_content=zone_content, metadata={"source": "speed_limits.json", "doc_type": "speed_zones"})
    )

    for state, note in data.get("state_highway_notes", {}).items():
        docs.append(
            Document(
                page_content=f"{state}: {note}",
                metadata={"source": "speed_limits.json", "state": state, "doc_type": "speed_zones"},
            )
        )

    logger.info(f"Parsed {len(docs)} speed limit entries from {path}")
    return docs


def _markdown_to_documents(path: Path, splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """Chunk a free-text markdown file (e.g. the traffic FAQ).

    Args:
        path: Path to the markdown file.
        splitter: A configured text splitter.

    Returns:
        List of chunked LangChain Documents.
    """
    text = path.read_text()
    chunks = splitter.split_text(text)
    docs = [
        Document(page_content=chunk, metadata={"source": path.name, "doc_type": "faq", "chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]
    logger.info(f"Chunked {path.name} into {len(docs)} pieces")
    return docs


def build_documents(kb_dir: Path = KNOWLEDGE_BASE_DIR) -> List[Document]:
    """Load every known knowledge-base source file into Documents.

    Args:
        kb_dir: Directory containing the knowledge base source files.

    Returns:
        Combined list of all Documents ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP, separators=["\n\n", "\n", ". ", " "]
    )

    documents: List[Document] = []

    sections_path = kb_dir / "mv_act_sections.json"
    if sections_path.exists():
        documents.extend(_mv_act_sections_to_documents(sections_path))

    fines_path = kb_dir / "fine_schedule.json"
    if fines_path.exists():
        documents.extend(_fine_schedule_to_documents(fines_path))

    speed_path = kb_dir / "speed_limits.json"
    if speed_path.exists():
        documents.extend(_speed_limits_to_documents(speed_path))

    faq_path = kb_dir / "traffic_faq.md"
    if faq_path.exists():
        documents.extend(_markdown_to_documents(faq_path, splitter))

    # Also pick up any additional .md/.txt reference docs dropped into the folder.
    for extra_path in list(kb_dir.glob("*.md")) + list(kb_dir.glob("*.txt")):
        if extra_path.name == faq_path.name:
            continue
        documents.extend(_markdown_to_documents(extra_path, splitter))

    if not documents:
        logger.info(f"No knowledge base source files found under {kb_dir}")
    return documents


def document_id(doc: Document) -> str:
    """Stable id for a Document (hash of source + text), so re-ingesting updates instead of duplicating."""
    key = f"{doc.metadata.get('source', '')}|{doc.page_content}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def ingest(rebuild: bool = False) -> int:
    """Embed all knowledge-base Documents into the persistent Chroma store.

    Args:
        rebuild: If True, delete the existing collection first so this
            run produces a clean re-embed (useful after editing the
            knowledge base source files).

    Returns:
        Number of documents ingested.
    """
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

    documents = build_documents()
    if not documents:
        logger.info("Nothing to ingest.")
        return 0

    # Local sentence-transformer embeddings: no API key required for
    # ingestion, keeping the RAG index buildable offline / in CI.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    CHROMADB_STORE_DIR.mkdir(parents=True, exist_ok=True)

    if rebuild:
        import shutil

        shutil.rmtree(CHROMADB_STORE_DIR, ignore_errors=True)
        CHROMADB_STORE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Rebuild requested: wiped existing ChromaDB store.")

    vectorstore = Chroma(
        collection_name=RAG_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMADB_STORE_DIR),
    )
    # Same documents always get the same ids: running ingest twice no longer doubles the collection
    # (duplicates would crowd real matches out of the top-k).
    vectorstore.add_documents(documents, ids=[document_id(d) for d in documents])
    logger.info(f"Ingested {len(documents)} documents into collection '{RAG_COLLECTION_NAME}'")
    return len(documents)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Ingest DriveGuard AI's knowledge base into ChromaDB.")
    parser.add_argument("--rebuild", action="store_true", help="Wipe and re-embed the entire collection.")
    args = parser.parse_args()

    count = ingest(rebuild=args.rebuild)
    print(f"Ingested {count} documents into '{RAG_COLLECTION_NAME}' at {CHROMADB_STORE_DIR}")


if __name__ == "__main__":
    main()
