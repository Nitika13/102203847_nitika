from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")  # small and fast

def get_text_embedding(text: str):
    vec = model.encode(text, show_progress_bar=False)
    return vec.tolist()
