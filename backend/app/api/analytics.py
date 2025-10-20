# backend/app/api/analytics.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
from typing import List, Dict

router = APIRouter()

DATA_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "products_with_text.csv"),
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "products.csv"),
]

def _load_products_df() -> pd.DataFrame:
    """Load CSV from known locations. Return empty DataFrame if none found."""
    for p in DATA_CANDIDATES:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                return df
            except Exception:
                continue
    # fallback: empty DF
    return pd.DataFrame()

@router.get("/summary", summary="Product dataset summary")
def summary():
    """
    Returns basic analytics summary:
      - total_products
      - avg_price (safe)
      - top_brands (list of {brand, count})
      - top_categories (list of {category, count})
    This function is defensive: converts price to numeric with errors='coerce'
    and handles missing columns.
    """
    try:
        df = _load_products_df()
        if df is None or df.empty:
            return {
                "total_products": 0,
                "avg_price": None,
                "top_brands": [],
                "top_categories": []
            }

        # total products
        total_products = int(len(df))

        # safe numeric conversion for price-like columns
        price = None
        if "price" in df.columns:
            # coerce invalid values to NaN, then compute mean ignoring NaNs
            price_series = pd.to_numeric(df["price"], errors="coerce")
            if price_series.dropna().size > 0:
                avg_price_val = float(price_series.mean())
                # round to 2 decimals
                price = round(avg_price_val, 2)
            else:
                price = None

        # top brands (if column exists)
        top_brands: List[Dict] = []
        if "brand" in df.columns:
            brand_counts = df["brand"].fillna("Unknown").astype(str).value_counts().head(10)
            top_brands = [{"brand": str(idx), "count": int(cnt)} for idx, cnt in brand_counts.items()]

        # top categories (try 'categories' or 'category' or 'type')
        cat_col = None
        for c in ["categories", "category", "type"]:
            if c in df.columns:
                cat_col = c
                break
        top_categories: List[Dict] = []
        if cat_col:
            cat_counts = df[cat_col].fillna("Unknown").astype(str).value_counts().head(10)
            top_categories = [{"category": str(idx), "count": int(cnt)} for idx, cnt in cat_counts.items()]

        return {
            "total_products": total_products,
            "avg_price": price,
            "top_brands": top_brands,
            "top_categories": top_categories
        }
    except Exception as e:
        # defensive: never allow backend import-time crashes from analytics
        raise HTTPException(status_code=500, detail=f"Analytics error: {e}")
