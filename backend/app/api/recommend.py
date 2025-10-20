# backend/app/api/recommend.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.utils.embeddings import get_text_embedding
from app.utils.faiss_client import init_index, query_vector
from app.api.genai import generate_description
import math

router = APIRouter()

class Prompt(BaseModel):
    prompt: str = Field(...)
    k: int = Field(6)

def _safe_float(x):
    """Return a Python float or None for invalid inputs (strings, NaN, inf)."""
    try:
        if x is None:
            return None
        # if already float or int
        if isinstance(x, (float, int)):
            if math.isnan(float(x)) or math.isinf(float(x)):
                return None
            return float(x)
        # try numeric conversion from string
        v = float(str(x).replace(",", "").strip())
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

@router.post("/", summary="Get product recommendations")
async def recommend(prompt: Prompt):
    text = (prompt.prompt or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    # generate query embedding
    q_emb = get_text_embedding(text)
    if q_emb is None:
        raise HTTPException(status_code=500, detail="Failed to generate embedding.")

    # ensure index initialized for safety
    init_index(len(q_emb))

    raw_results = query_vector(q_emb, top_k=max(20, prompt.k*3))  # fetch more for dedupe/rerank
    if not raw_results:
        return {"results": [], "message": "No items indexed yet. Upload dataset first."}

    # Deduplicate results by id, keep highest score per id
    best_by_id = {}
    for r in raw_results:
        rid = str(r.get("id"))
        score = r.get("score", 0.0)
        try:
            score = float(score)
        except Exception:
            score = 0.0
        if rid not in best_by_id or score > best_by_id[rid]['score']:
            best_by_id[rid] = {"id": rid, "score": score, "metadata": r.get("metadata", {})}

    # convert to list and sort by score desc, then pick top-k
    deduped = sorted(best_by_id.values(), key=lambda x: x['score'], reverse=True)[:prompt.k]

    enriched = []
    for r in deduped:
        meta = r.get("metadata", {}) or {}
        score = r.get("score", 0.0)
        if score is None or (isinstance(score, float) and (math.isnan(score) or math.isinf(score))):
            score = 0.0

        # sanitize price: convert to float or set None
        price_val = _safe_float(meta.get("price") if meta is not None else None)
        # prepare price string for description
        price_str = f"₹{int(price_val)}" if (price_val is not None and price_val == int(price_val)) else (f"₹{round(price_val,2)}" if price_val is not None else "N/A")

        # generate a safe description using your genai helper (it should already be defensive)
        # pass price_str explicitly so genai can use it without reading raw meta
        try:
            description = generate_description(meta, text)
            # if description contains 'nan' replace it (defensive)
            if isinstance(description, str) and "nan" in description:
                description = description.replace("nan", "N/A")
        except Exception:
            description = f"{meta.get('title', 'Product')} — priced at {price_str}. Suitable for {text}."

        enriched.append({
            "id": r["id"],
            "score": round(float(score), 6),
            "metadata": meta,
            "price": price_val,
            "price_display": price_str,
            "generated_description": description
        })

    # final sanitize (ensure no JSON-invalid floats)
    def _sanitize(obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return 0.0
            return obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    return _sanitize({"results": enriched})
