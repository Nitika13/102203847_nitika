from fastapi import APIRouter
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from app.utils.faiss_client import save_index
import faiss

router = APIRouter()
model = SentenceTransformer("all-MiniLM-L6-v2")

@router.post("/")
def ingest_data():
    df = pd.read_csv("app/data/products.csv")
    embeddings = model.encode(df["product_name"].tolist())
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    metadata = df.to_dict(orient="records")
    save_index(index, metadata)

    return {"message": f"Indexed {len(df)} products locally."}
