"""
NeuroAssist AI v2 — RAG Service
Retrieves relevant clinical guideline chunks from the FAISS vector index.
"""

import os
from loguru import logger
from backend.config import settings


_index = None
_documents = None


def load_rag_index():
    """Load the FAISS index and document store at startup."""
    global _index, _documents

    index_path = os.path.join(settings.faiss_index_dir, "index.faiss")
    docs_path = os.path.join(settings.faiss_index_dir, "documents.json")

    if not os.path.exists(index_path):
        logger.warning(f"FAISS index not found at {index_path}. RAG will be unavailable.")
        return False

    try:
        import faiss
        import json

        _index = faiss.read_index(index_path)

        with open(docs_path, "r", encoding="utf-8") as f:
            _documents = json.load(f)

        logger.info(f"RAG index loaded: {_index.ntotal} vectors, {len(_documents)} documents")
        return True

    except Exception as e:
        logger.error(f"Failed to load RAG index: {e}")
        return False


def retrieve_relevant_guidelines(query: str, top_k: int = None) -> list:
    """
    Retrieve top-k relevant guideline chunks for a query.
    Returns list of {text, source, url} dicts.
    """
    if _index is None or _documents is None:
        logger.warning("RAG index not loaded. Returning empty results.")
        return []

    top_k = top_k or settings.rag_top_k

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        embedder = SentenceTransformer(settings.embedding_model)
        query_embedding = embedder.encode([query], convert_to_numpy=True)

        distances, indices = _index.search(query_embedding.astype("float32"), top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(_documents) and idx >= 0:
                doc = _documents[idx]
                results.append({
                    "text": doc.get("text", ""),
                    "source": doc.get("source", "Unknown"),
                    "url": doc.get("url", ""),
                    "score": float(distances[0][i]),
                })

        return results

    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}")
        return []
