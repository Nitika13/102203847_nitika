import faiss
import numpy as np
import os
import pickle

INDEX_PATH = "app/data/faiss_index.bin"
META_PATH = "app/data/metadata.pkl"

# Save and load helper functions
def save_index(index, metadata):
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

def load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    return None, None
