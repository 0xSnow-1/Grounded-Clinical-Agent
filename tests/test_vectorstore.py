"""Naive RAG vector store roundtrip test.

Verifies the full dense-only path with a LOCAL embedding model
(abhinand/MedEmbed-small-v0.1) and a throwaway Qdrant dir -- no AWS
tokens, no cloud endpoint, no touching the real clinical_guidelines
collection in data/qdrant_db.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from langchain_core.documents import Document

from rag.vectorstore import build_vectorstore, load_vectorstore

# A tiny stand-in corpus: two clinical chunks + one clearly unrelated chunk.
CHUNKS = [
    Document(
        page_content=(
            "Amoxicillin 2 g orally should be administered 30-60 minutes "
            "before the dental procedure for infective endocarditis "
            "antibiotic prophylaxis."
        ),
        metadata={"source": "AHA_ADA_Antibiotic_Prophylaxis_Guidelines.md"},
    ),
    Document(
        page_content=(
            "Fluoride varnish should be applied to the primary teeth of all "
            "infants and children starting at the age of primary tooth eruption."
        ),
        metadata={"source": "uspstf-oral-health-children.md"},
    ),
    Document(
        page_content="Chocolate cake is a dessert enjoyed by many people.",
        metadata={"source": "unrelated.txt"},
    ),
]


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Point the module at a throwaway Qdrant dir + collection name so the
    real data/qdrant_db/clinical_guidelines collection is never touched."""
    monkeypatch.setattr(
        "rag.vectorstore.QDRANT_PATH", str(tmp_path / "qdrant_db")
    )
    monkeypatch.setattr("rag.vectorstore.COLLECTION_NAME", "naive_smoke_test")
    return tmp_path


def test_build_load_search_roundtrip(isolated_store):
    # 1. Build path: embed chunks -> persist to local Qdrant.
    build_vectorstore(CHUNKS)

    # 2. Load path: reconnect to the existing collection WITHOUT re-embedding.
    store = load_vectorstore()

    # 3. Search path: nearest chunks for a clinical query (dense similarity).
    hits = store.similarity_search(
        "What amoxicillin dose is recommended before dental procedures?",
        k=2,
    )

    assert len(hits) == 2
    assert "Amoxicillin" in hits[0].page_content
    assert (
        hits[0].metadata["source"]
        == "AHA_ADA_Antibiotic_Prophylaxis_Guidelines.md"
    )

    # The unrelated dessert chunk must NOT be the top hit for a clinical query.
    assert hits[0].metadata["source"] != "unrelated.txt"


def test_search_with_scores_returns_ordered_results(isolated_store):
    build_vectorstore(CHUNKS)
    store = load_vectorstore()

    scored = store.similarity_search_with_score(
        "fluoride varnish application for children",
        k=2,
    )

    assert len(scored) == 2
    # Each entry is (Document, cosine-similarity-score in [-1, 1]).
    for doc, score in scored:
        assert 0.0 <= score <= 1.0
        assert isinstance(doc, Document)

    # Higher score first.
    assert scored[0][1] >= scored[1][1]
    assert "Fluoride varnish" in scored[0][0].page_content