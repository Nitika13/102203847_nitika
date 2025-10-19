# backend/app/api/analytics.py
from fastapi import APIRouter, HTTPException
import pandas as pd
import os

router = APIRouter()

@router.get("/summary", summary="Get analytics summary from products.csv")
def summary():
    # find the data path
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(base_dir, "data", "products.csv")

    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail=f"File not found: {data_path}")

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    # compute analytics
    cat_counts = df['categories'].fillna("unknown").value_counts().to_dict()

    price_stats = {
        "mean": float(df['price'].mean()),
        "median": float(df['price'].median()),
        "min": float(df['price'].min()),
        "max": float(df['price'].max())
    }

    return {"categories": cat_counts, "price": price_stats}
