"""End-to-end HYBRID (dense + sparse + RRF) vector store tests.

Uses a throwaway Qdrant dir + collection name so the real
data/qdrant_db/clinical_guidelines collection is never touched, and the
local MedEmbed + Qdrant/bm25 models -- no AWS tokens.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from rag import vectorstore
from rag.retrieval import hybrid_retrieve_chunks

# Small corpus with clear lexical + semantic signals:
# two clinical chunks, one clearly unrelated.
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
def isolated_hybrid(tmp_path, monkeypatch):
    """Point both modules at a throwaway Qdrant dir + collection name."""
    test_path = str(tmp_path / "qdrant_db")
    test_collection = "hybrid_smoke_test"
    monkeypatch.setattr(vectorstore, "QDRANT_PATH", test_path)
    monkeypatch.setattr(vectorstore, "COLLECTION_NAME", test_collection)
    # The module-level client was built against the REAL path at import time;
    # rebind it to the test dir so collection management + queries go to tmp.
    monkeypatch.setattr(
        vectorstore, "qdrant_client", QdrantClient(path=test_path)
    )
    return test_collection


def test_build_creates_hybrid_collection(isolated_hybrid):
    vectorstore.build_vectorstore(CHUNKS)

    info = vectorstore.qdrant_client.get_collection(
        collection_name=vectorstore.COLLECTION_NAME
    )
    vectors = info.config.params.vectors
    assert isinstance(vectors, dict)
    assert set(vectors) == {"dense"}
    assert vectors["dense"].distance == "Cosine"
    assert vectorstore.SPARSE_VECTOR_NAME in info.config.params.sparse_vectors
    sparse_params = info.config.params.sparse_vectors["sparse"]
    assert sparse_params.modifier.value == "idf"

    assert info.points_count == len(CHUNKS)


def test_hybrid_retrieval_returns_relevant_chunk(isolated_hybrid):
    vectorstore.build_vectorstore(CHUNKS)

    hits = hybrid_retrieve_chunks(
        "recommended amoxicillin dose before dental procedures", k=3
    )

    assert len(hits) == 3
    top_doc, top_score = hits[0]

    # Sparse BM25 matches on the exact drug name; dense matches on meaning.
    assert "Amoxicillin" in top_doc.page_content
    assert (
        top_doc.metadata["source"]
        == "AHA_ADA_Antibiotic_Prophylaxis_Guidelines.md"
    )
    assert top_score > 0  # RRF fused score is positive
    # Scores should be descending (best first).
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_sparse_alone_finds_lexical_match(isolated_hybrid):
    """The sparse side alone should surface the chunk with the exact drug
    token, demonstrating the hybrid signal actually comes from both sides."""
    vectorstore.build_vectorstore(CHUNKS)

    emb = next(vectorstore.sparse_model.query_embed("amoxicillin"))
    sparse_vec = vectorstore.models.SparseVector(
        indices=list(emb.as_dict().keys()),
        values=list(emb.as_dict().values()),
    )
    resp = vectorstore.qdrant_client.query_points(
        collection_name=vectorstore.COLLECTION_NAME,
        query=sparse_vec,
        using=vectorstore.SPARSE_VECTOR_NAME,
        limit=1,
        with_payload=True,
    )

    assert resp.points[0].payload["metadata"]["source"].startswith("AHA_ADA")


def test_dense_fallback_still_works(isolated_hybrid):
    """build_vectorstore returns a LangChain store bound to the dense vector;
    plain similarity_search must keep working for the naive path."""
    store = vectorstore.build_vectorstore(CHUNKS)

    hits = store.similarity_search(
        "What amoxicillin dose is recommended before dental procedures?",
        k=2,
    )

    assert len(hits) == 2
    assert "Amoxicillin" in hits[0].page_content
    assert hits[0].metadata["source"] == "AHA_ADA_Antibiotic_Prophylaxis_Guidelines.md"