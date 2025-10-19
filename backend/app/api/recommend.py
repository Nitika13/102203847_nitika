from fastapi import APIRouter
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from app.utils.faiss_client import load_index

router = APIRouter()
model = SentenceTransformer("all-MiniLM-L6-v2")

class Query(BaseModel):
    query: str

@router.post("/")  
def recommend(query: Query):
    index, metadata = load_index()
    if index is None:
        return {"message": "No items indexed yet. Please upload dataset first.", "results": []}

    query_emb = np.array([model.encode(query.query)]).astype("float32")
    D, I = index.search(query_emb, 5)
    results = [metadata[i] for i in I[0]]

    return {"results": results}
