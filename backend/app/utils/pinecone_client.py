# backend/app/utils/pinecone_client.py
import numpy as np

_LOCAL_STORE = {}

def upsert_items_local(items):
    """Save items to a simple in-memory dictionary."""
    for it in items:
        _LOCAL_STORE[it["id"]] = {
            "embedding": np.array(it["embedding"]),
            "metadata": it["metadata"]
        }

def query_vector_local(embedding, top_k=6):
    """Return top-k most similar items based on cosine similarity."""
    if len(_LOCAL_STORE) == 0:
        return []

    emb = np.array(embedding)
    results = []

    def safe_cosine(a, b):
        # compute cosine similarity safely (avoid divide-by-zero)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        score = float(np.dot(a, b) / denom)
        if np.isnan(score) or np.isinf(score):
            score = 0.0
        return score

    for _id, val in _LOCAL_STORE.items():
        score = safe_cosine(emb, val["embedding"])
        results.append((_id, score, val["metadata"]))

    # sort and pick top_k
    results.sort(key=lambda x: x[1], reverse=True)
    top = []
    for sid, score, meta in results[:top_k]:
        if np.isnan(score) or np.isinf(score):
            score = 0.0
        top.append({"id": sid, "score": float(score), "metadata": meta})

    return top
