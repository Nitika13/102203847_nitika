# backend/app/api/ingest.py
from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from app.utils.embeddings import get_text_embedding
from app.utils.faiss_client import init_index, upsert_items
from typing import List, Dict

router = APIRouter()

@router.post("/dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV file (products.csv). CSV must contain at least title/description (or text fields).
    This endpoint computes embeddings and upserts them into the FAISS index.
    """
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {e}")

    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty or invalid.")

    # build sample embedding to determine dimension
    # use first non-empty title/description or fallback to first row
    sample_row = df.iloc[0]
    sample_text = f"{sample_row.get('title','')} {sample_row.get('description','')}".strip()
    if not sample_text:
        # fallback to whole row string
        sample_text = " ".join([str(v) for v in sample_row.values if pd.notna(v)])
    sample_emb = get_text_embedding(sample_text)
    if sample_emb is None:
        raise HTTPException(status_code=500, detail="Failed to generate sample embedding.")

    dim = len(sample_emb)
    # initialize index with dimension
    init_index(dim)

    items = []
    for idx, row in df.iterrows():
        try:
            text = f"{row.get('title','')} . {row.get('description','')} . {row.get('categories','')}"
            emb = get_text_embedding(text)
            if emb is None:
                # skip rows where embedding failed
                continue
            meta = row.to_dict()
            items.append({"id": str(row.get('uniq_id', idx)), "embedding": emb, "metadata": meta})
        except Exception:
            # skip malformed rows but continue indexing the rest
            continue

    if len(items) == 0:
        raise HTTPException(status_code=500, detail="No items to index (all embeddings failed).")

    # upsert all items into FAISS (faiss_client handles persistence)
    upsert_items(items)

    return {"status": "indexed", "count": len(items)}
