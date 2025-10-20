# backend/app/utils/faiss_client.py
import os
import json
import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
os.makedirs(BASE_DIR, exist_ok=True)
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.idx")
EMB_PATH = os.path.join(BASE_DIR, "faiss_embeddings.npy")
META_PATH = os.path.join(BASE_DIR, "faiss_metadata.json")
IDX_MAP_PATH = os.path.join(BASE_DIR, "faiss_index_map.json")

# in-memory containers
_faiss_index = None
_embeddings = None    # numpy array (n x dim)
_metadata = {}        # id -> metadata dict
_index_map = []       # list of ids in the same order as embeddings
_dim = None

def _save_metadata():
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(_metadata, f, ensure_ascii=False)

def _load_metadata():
    global _metadata, _index_map
    _metadata = {}
    _index_map = []
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
    if os.path.exists(IDX_MAP_PATH):
        with open(IDX_MAP_PATH, "r", encoding="utf-8") as f:
            _index_map = json.load(f)
    else:
        # derive from metadata order as fallback
        _index_map = list(_metadata.keys())

def init_index(dim:int):
    """
    Initialize or load index. Call before upsert/query.
    """
    global _faiss_index, _embeddings, _dim
    _dim = int(dim)
    _load_metadata()
    if _HAS_FAISS:
        if os.path.exists(INDEX_PATH) and os.path.exists(EMB_PATH):
            try:
                _faiss_index = faiss.read_index(INDEX_PATH)
                _embeddings = np.load(EMB_PATH)
            except Exception:
                # start fresh if load fails
                _faiss_index = faiss.IndexFlatIP(_dim)
                _embeddings = np.zeros((0, _dim), dtype="float32")
        else:
            _faiss_index = faiss.IndexFlatIP(_dim)
            _embeddings = np.zeros((0, _dim), dtype="float32")
    else:
        if os.path.exists(EMB_PATH):
            _embeddings = np.load(EMB_PATH)
        else:
            _embeddings = np.zeros((0, _dim), dtype="float32")
        _faiss_index = None

def upsert_items(items):
    """
    items: list of dicts {"id": str, "embedding": array-like, "metadata": dict}
    """
    global _faiss_index, _embeddings, _metadata, _index_map, _dim
    if _dim is None:
        raise RuntimeError("Call init_index(dim) before upsert_items")

    rows = []
    ids_to_add = []
    for it in items:
        iid = str(it["id"])
        emb = np.array(it["embedding"], dtype="float32").reshape(-1)
        if emb.size != _dim:
            # skip mismatched dims
            continue
        # safe normalize
        norm = np.linalg.norm(emb)
        if norm == 0 or np.isnan(norm) or np.isinf(norm):
            emb = np.zeros_like(emb, dtype="float32")
        else:
            emb = (emb / norm).astype("float32")
        rows.append(emb)
        ids_to_add.append(iid)
        _metadata[iid] = it.get("metadata", {})

    if len(rows) == 0:
        return

    vecs = np.vstack(rows).astype("float32")

    # append
    if _HAS_FAISS:
        _faiss_index.add(vecs)
    # append to embeddings array
    if _embeddings is None or _embeddings.size == 0:
        _embeddings = vecs
    else:
        _embeddings = np.vstack([_embeddings, vecs])

    # update index map in exact insertion order
    _index_map.extend(ids_to_add)
    # persist
    if _HAS_FAISS:
        faiss.write_index(_faiss_index, INDEX_PATH)
    np.save(EMB_PATH, _embeddings)
    _save_metadata()
    with open(IDX_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(_index_map, f, ensure_ascii=False)

def query_vector(query_embedding, top_k=6):
    """
    Returns list of {"id", "score", "metadata"} sorted by descending score.
    Always returns finite floats (no NaN/inf).
    """
    global _faiss_index, _embeddings, _metadata, _index_map
    if _embeddings is None or _embeddings.size == 0:
        return []

    q = np.array(query_embedding, dtype="float32").reshape(1, -1)
    qnorm = np.linalg.norm(q)
    if qnorm == 0 or np.isnan(qnorm) or np.isinf(qnorm):
        q = np.zeros_like(q)
    else:
        q = q / qnorm

    results = []
    if _HAS_FAISS:
        D, I = _faiss_index.search(q, top_k)
        D = D.astype("float32")
        I = I.astype(int)
        # ensure index_map loaded
        if not _index_map:
            _load_metadata()
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            # map idx -> id safely
            if idx < len(_index_map):
                iid = _index_map[idx]
            else:
                iid = str(idx)
            s = float(score)
            if np.isnan(s) or np.isinf(s):
                s = 0.0
            results.append({"id": iid, "score": s, "metadata": _metadata.get(iid, {})})
    else:
        # fallback: cosine via dot product (embeddings already normalized)
        sims = (_embeddings @ q.T).reshape(-1)
        top_idx = np.argsort(sims)[-top_k:][::-1]
        if not _index_map:
            _load_metadata()
        for idx in top_idx:
            s = float(sims[idx])
            if np.isnan(s) or np.isinf(s):
                s = 0.0
            if idx < len(_index_map):
                iid = _index_map[idx]
            else:
                iid = str(idx)
            results.append({"id": iid, "score": s, "metadata": _metadata.get(iid, {})})
    return results
