# quick script to print recall results
import joblib, os, numpy as np
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
res = joblib.load(os.path.join(OUT_DIR, "evaluation_results.pkl"))
print(res)
