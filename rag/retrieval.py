















# """
# Retrieval step: query -> most relevant chunks.

# Dense-only path (v0): plain similarity search via the LangChain store.
# Hybrid path: low-level Qdrant query_points with a dense prefetch + a sparse
# (BM25) prefetch, fused with Reciprocal Rank Fusion (RRF).
# """
# import threading

# from langchain_core.documents import Document

# from qdrant_client import models

# from . import vectorstore
# from .vectorstore import load_vectorstore

# # Loaded once and reused, not reopened on every call -- important once this
# # is invoked repeatedly inside an agent loop.
# _store = None
# _store_lock = threading.Lock()


# def _get_store():
#     global _store
#     if _store is None:
#         with _store_lock:
#             if _store is None:
#                 _store = load_vectorstore()
#     return _store


# # --------------------------------------------------------------------------
# # Dense-only (naive) retrieval
# # --------------------------------------------------------------------------

# def retrieve_chunks(query: str, k: int = 3) -> list[Document]:
#     """Embeds the query with the same model used at ingestion time, and
#     returns the k closest chunks by similarity.
#     """
#     return _get_store().similarity_search(query=query, k=k)


# def retrieve_chunks_with_scores(query: str, k: int = 3) -> list[tuple[Document, float]]:
#     """Same as retrieve_chunks, but also returns each chunk's similarity
#     score -- useful when the caller (e.g. an agent) needs to judge confidence
#     rather than blindly trusting the top-k.
#     """
#     return _get_store().similarity_search_with_score(query=query, k=k)


# # --------------------------------------------------------------------------
# # Hybrid (dense + sparse, RRF-fused) retrieval
# # --------------------------------------------------------------------------

# def _query_dense(query: str) -> list[float]:
#     return vectorstore.embedding_model.embed_query(query)


# def _query_sparse(query: str) -> models.SparseVector:
#     emb = next(vectorstore.sparse_model.query_embed(query))
#     return models.SparseVector(
#         indices=list(emb.as_dict().keys()),
#         values=list(emb.as_dict().values()),
#     )


# def hybrid_retrieve_chunks(query: str, k: int = 3) -> list[tuple[Document, float]]:
#     """Hybrid retrieval: run a dense prefetch and a sparse (BM25) prefetch on
#     the named "dense"/"sparse" vector spaces, then fuse the two candidate
#     pools with Reciprocal Rank Fusion.

#     Returns (Document, fused_score) pairs -- the fused score is a rank-based
#     score from RRF, not a raw cosine similarity.
#     """
#     prefetch = [
#         models.Prefetch(
#             query=_query_dense(query),
#             using=vectorstore.DENSE_VECTOR_NAME,
#             limit=k * 4,  # oversample each pool before fusion
#         ),
#         models.Prefetch(
#             query=_query_sparse(query),
#             using=vectorstore.SPARSE_VECTOR_NAME,
#             limit=k * 4,
#         ),
#     ]

#     response = vectorstore.qdrant_client.query_points(
#         collection_name=vectorstore.COLLECTION_NAME,
#         prefetch=prefetch,
#         query=models.FusionQuery(fusion=models.Fusion.RRF),
#         limit=k,
#         with_payload=True,
#     )

#     results = []
#     for point in response.points:
#         payload = point.payload or {}
#         results.append(
#             (
#                 Document(
#                     page_content=payload.get("page_content", ""),
#                     metadata=payload.get("metadata", {}),
#                 ),
#                 point.score,
#             )
#         )
#     return results


# if __name__ == "__main__":
#     test_query = "What amoxicillin dose is recommended before dental procedures?"
#     print(f"Query: {test_query}\n")

#     print("=== Dense-only (naive) ===")
#     for i, doc in enumerate(retrieve_chunks(test_query, k=3), start=1):
#         print(f"--- result {i} (source: {doc.metadata.get('source')}) ---")
#         print(doc.page_content[:200].replace("\n", " "))
#         print()

#     print("=== Hybrid (dense + BM25, RRF) ===")
#     for i, (doc, score) in enumerate(hybrid_retrieve_chunks(test_query, k=3), start=1):
#         print(f"--- result {i} (fused score: {score:.4f}) ---")
#         print(f"source: {doc.metadata.get('source')}")
#         print(doc.page_content[:200].replace("\n", " "))
#         print()
