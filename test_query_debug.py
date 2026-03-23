
import numpy as np
import os
import sys

# Add current dir to path
sys.path.append(os.getcwd())

from app.embeddings import encode_texts, load_from_disk
from app.search import VectorSearch

try:
    print("Loading data...")
    idx, embs, docs = load_from_disk()
    if idx is None:
        print("No index found.")
        sys.exit(1)
        
    print(f"Loaded {len(docs)} documents.")
    vs = VectorSearch(embs, docs, index=idx)
    
    query = "Photosynthesis"
    print(f"Encoding query: {query}")
    q_emb = encode_texts([query])[0]
    
    print("Searching...")
    # Convert query embedding to float32 before passing
    q_emb_array = np.array([q_emb]).astype("float32")
    results = vs.search(q_emb_array, query_text=query, k=6)
    
    print(f"Found {len(results['results'])} results.")
    for i, r in enumerate(results['results']):
        print(f"{i+1}. Class {r['class']} - {r['chapter']} (Score: {r['score']})")
        
except Exception as e:
    import traceback
    traceback.print_exc()
