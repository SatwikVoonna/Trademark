"""
FastAPI Service — SciSearch (NCERT Science Semantic Search)

Endpoints:
- POST /query — semantic search returning { class, chapter, passage, score, subject, keywords }
- GET /cache/stats — cache statistics
- DELETE /cache — flush cache
- Static frontend served from /static and / root
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import os

from app.embeddings import (
    load_documents,
    generate_embeddings,
    encode_texts,
    save_to_disk,
    load_from_disk,
    DATASET_DIR,
)
from app.search import VectorSearch
from app.clustering import FuzzyCluster
from app.cache import SemanticCache


app = FastAPI(
    title="SciSearch — NCERT Science Semantic Search",
    description="Semantic search across NCERT Science textbooks (Class 6–11)",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup: Load or generate embeddings
# ---------------------------------------------------------------------------

print("Checking for persisted data...")
index, embeddings, documents = load_from_disk()

if index is not None:
    print(f"Loaded persisted data from disk ({len(documents)} passages).")
    embeddings = np.array(embeddings).astype("float32")
else:
    print("No persisted data found. Processing NCERT PDFs...")
    documents = load_documents()
    print(f"Loaded {len(documents)} passages after processing.")

    print("Generating embeddings (this may take a few minutes)...")
    embeddings = generate_embeddings(documents)
    embeddings = np.array(embeddings).astype("float32")

    print("Persisting to disk...")
    save_to_disk(embeddings, documents)
    index = None

# Build vector search engine
if index is not None:
    vector_search = VectorSearch(embeddings, documents, index=index)
else:
    vector_search = VectorSearch(embeddings, documents)

# ---------------------------------------------------------------------------
# Fuzzy Clustering
# ---------------------------------------------------------------------------
optimal_k = 8
print(f"\nTraining fuzzy cluster model with k={optimal_k}...")
cluster_model = FuzzyCluster(n_clusters=optimal_k)
cluster_model.train(embeddings)

# Print brief cluster analysis
print("\nCluster analysis:")
analysis = cluster_model.analyze_clusters(embeddings, documents, n_examples=2)
for c in analysis["clusters"]:
    print(f"  Cluster {c['cluster_id']}: {c['size']} documents")

# ---------------------------------------------------------------------------
# Semantic Cache
# ---------------------------------------------------------------------------
cache = SemanticCache(cluster_model=cluster_model, threshold=0.85)


# ---------------------------------------------------------------------------
# API Models & Endpoints
# ---------------------------------------------------------------------------

class Query(BaseModel):
    query: str
    filters: dict = {}


@app.post("/query")
def query_endpoint(q: Query):
    """
    Semantic search endpoint.

    Returns top 6 results with:
    { class, chapter, passage, score, subject, keywords }
    """
    query_embedding = encode_texts([q.query])[0]

    # Cache lookup
    hit, entry, sim = cache.lookup(query_embedding, q.query, q.filters)

    if hit:
        return {
            "query": q.query,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": float(sim),
            "result": entry["result"]["result"],
            "facets": entry["result"].get("facets", {}),
        }

    # Cache miss: compute fresh result
    search_output = vector_search.search(
        np.array([query_embedding]).astype("float32"),
        query_text=q.query,
        k=6,
        filters=q.filters,
    )
    results = search_output["results"]
    facets = search_output["facets"]

    dominant_cluster, cluster_probs = cluster_model.predict(
        np.array([query_embedding])
    )

    response = {
        "query": q.query,
        "cache_hit": False,
        "result": results,
        "facets": facets,
        "dominant_cluster": int(dominant_cluster),
    }

    # Store in cache
    cache.store(query_embedding, q.query, response, dominant_cluster, q.filters)

    return response


@app.get("/cache/stats")
def stats_endpoint():
    """Return current cache statistics."""
    return cache.stats()


@app.delete("/cache")
def clear_endpoint():
    """Flush the cache entirely."""
    cache.clear()
    return {"message": "cache cleared"}


# ---------------------------------------------------------------------------
# Serve Frontend
# ---------------------------------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.get("/")
def serve_index():
    """Serve the main frontend page."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# Mount static files (CSS, JS) — must come AFTER route definitions
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Mount PDF files directory
if os.path.exists(DATASET_DIR):
    app.mount("/pdfs", StaticFiles(directory=DATASET_DIR), name="pdfs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)