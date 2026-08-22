"""
Embedding + storage step: chunks -> persisted vector store.

HYBRID RAG: each chunk is stored with BOTH a dense vector (MedEmbed-small-v0.1)
and a sparse vector (Qdrant/bm25, IDF-modulated) in a single Qdrant collection
with named vector spaces ("dense" and "sparse"). Retrieval then runs both
prefetches and fuses them with Reciprocal Rank Fusion (RRF) -- see
rag/retrieval.py::hybrid_retrieve_chunks.

The public API (build_vectorstore / load_vectorstore) is unchanged so
ingest.py and the dense-only fallback keep working. load_hybrid_vectorstore
is the explicit hybrid-aware entry point. All storage is local on-disk via
QdrantClient(path=...) unless QDRANT_URL/QDRANT_API_KEY are set.
"""
from pathlib import Path
import os
from typing import Sequence

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
import uuid

load_dotenv()
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

REPO_ROOT = Path(__file__).resolve().parents[1]
QDRANT_PATH = str(REPO_ROOT / "data" / "qdrant_db")
COLLECTION_NAME = "clinical_guidelines_hybrid"

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
SPARSE_MODEL_NAME = "Qdrant/bm25"  # or "prithivida/Splade_PP_en_v1" later
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

# One dense model instance, reused by both build and load. Must be the SAME
# model at ingestion time and query time, otherwise the query vectors live in
# a different space than the stored document vectors and similarity search
# silently returns garbage.
dense_model = HuggingFaceEmbeddings(model_name="abhinand/MedEmbed-small-v0.1")


QDRANT_URL = os.getenv("QDRANT_URL") or os.getenv("CLUSTER_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or os.getenv("API_KEY")

client =  QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config= {
        "dense": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
    },
    sparse_vectors_config= {
        "sparse": models.SparseVectorParams(
            modifier=models.Modifier.IDF
        )
    }
)

print(f"Schema for '{COLLECTION_NAME}' created successfully.")

def upload_data(text_chunks, dense_vectors, sparse_vectors):
    payload_batch = []

    for i, text in enumerate(text_chunks):
        current_sparse = sparse_vectors[i]

        models.PointStruct(
            id= uuid.uuid4(),
            vector= {
                "dense": dense_vectors[i],
                "sparse": models.SparseVector(
                    indices=current_sparse.indices.tolist(),
                    values=current_sparse.values.tolst()
                )
            },
        payload= {"text":text}
        )

        client.upsert(collection_name=COLLECTION_NAME, points=payload_batch)





