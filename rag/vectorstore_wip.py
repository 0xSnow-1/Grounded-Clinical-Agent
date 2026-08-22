"""
Embedding + storage step: chunks -> persisted vector store.

v0 scope (NAIVE RAG): single dense embedding model stored in a local,
persisted Qdrant collection. No hybrid (sparse) vectors, no reranking,
no cloud endpoint -- everything runs locally so no API tokens are needed.

This is the baseline the hybrid-search upgrade builds on: when you
implement hybrid retrieval, you will extend build_vectorstore to also
configure a sparse model (e.g. Qdrant/bm25) on the collection, and extend
the search path in rag/retrieval.py to run dense + sparse queries and fuse
their results. Keep this file's public API intact so ingest.py and
retrieval.py keep working.
"""
from pathlib import Path
import os

from langchain_core import embeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from typing import Iterable, Sequence

from qdrant_client import QdrantClient, models
from qdrant_client.http import exceptions as qdrant_exceptions
from fastembed import SparseTextEmbedding

REPO_ROOT = Path(__file__).resolve().parents[1]
QDRANT_PATH = str(REPO_ROOT / "data" / "qdrant_db")
COLLECTION_NAME = "clinical_guidelines"

# One embedding model instance, reused by both build and load.
# Must be the SAME model at ingestion time and query time, otherwise the
# query vectors live in a different space than the stored document vectors
# and similarity search silently returns garbage.
SPARSE_MODEL_NAME = "Qdrant/bm25"
embedding_model = HuggingFaceEmbeddings(model_name="abhinand/MedEmbed-small-v0.1")
sparse_model= SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

def build_vectorstore(chunks: list[Document]) -> QdrantVectorStore:
    """Ingestion path: embed already-chunked Documents into a NEW, persisted
    Qdrant collection on disk. Run this once per corpus update, not on every
    startup -- it re-embeds everything each time it's called.
    """
    return QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
    )

def encode_dense_texts(texts: Sequence[str]) -> list[list[float]]:
    """
    Encode texts using the existing HuggingFaceEmbeddings model.
    Returns list of dense vectors (list of floats).
    """

    return embedding_model.embed_documents(list(texts))


def encode_sparse_texts(texts: Sequence[str]) -> list[models.SparseVector]:
    """
    Encode texts into sparse vectors (BM25) via FastEmbed.
    Returns list of Qdrant SparseVector dicts.
    """
    
    embeddings = list(sparse_model.embed(texts))
    
    return[
        models.SparseVector(
            indices=list(emb.keys()),
            values=list(emb.values())
        )


        for emb in (e.as_dict() for e in embeddings)
    ] 
    
    



def load_vectorstore() -> QdrantVectorStore:
    """Usage path: connect to an already-populated Qdrant collection
    without re-ingesting anything. This is what retrieval.py should call --
    it should never rebuild the index on every query.
    """
    return QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
    )


# QdrantClient defaults to remote localhost:6333, but this project's store
# is on-disk -- so without a URL it must open the local path instead.
QDRANT_URL = os.getenv("QDRANT_URL") or os.getenv("CLUSTER_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or os.getenv("API_KEY")

if QDRANT_URL:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)
else:
    client = QdrantClient(path=QDRANT_PATH)