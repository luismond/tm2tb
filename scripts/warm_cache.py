# scripts/warm_cache.py
import subprocess

# Preload sentence-transformers model
print("Downloading SentenceTransformer model…")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('LaBSE')
model.save('sbert_models/LaBSE')

print("✅ Caches warmed.")
