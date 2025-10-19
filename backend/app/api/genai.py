# backend/app/api/genai.py
def generate_description(metadata, user_prompt):
    """Simple safe description generator."""
    try:
        title = metadata.get("title") or "Product"
        brand = metadata.get("brand") or ""
        price = metadata.get("price") or ""
        base = f"{title}"
        if brand:
            base += f" by {brand}"
        if price:
            base += f" — priced at {price}"
        return f"{base}. Great choice when you want {user_prompt}."
    except Exception:
        # absolute fallback so it never breaks FastAPI
        return f"A product suitable for '{user_prompt}'."
