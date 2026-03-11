"""
FastAPI Service (Part 4)

This module wires together all components:
- Embedding model for encoding queries
- FAISS vector search for document retrieval
- GMM fuzzy clustering for soft topic assignment
- Cluster-accelerated semantic cache for deduplication

Startup behavior:
- If persisted data exists on disk, load it (fast restart).
- Otherwise, download the dataset, generate embeddings, and persist them.

Endpoints match the specification exactly:
- POST /query — semantic search with caching
- GET /cache/stats — cache statistics
- DELETE /cache — flush cache
"""

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from app.embeddings import (
    load_documents,
    generate_embeddings,
    model,
    save_to_disk,
    load_from_disk,
)
from app.search import VectorSearch
from app.clustering import FuzzyCluster
from app.cache import SemanticCache


app = FastAPI(
    title="Semantic Search System",
    description="Lightweight semantic search with fuzzy clustering and semantic caching",
)

# ---------------------------------------------------------------------------
# Startup: Load or generate embeddings, build search index and cluster model
# ---------------------------------------------------------------------------

print("Checking for persisted data...")
index, embeddings, documents = load_from_disk()

if index is not None:
    print("Loaded persisted data from disk.")
    embeddings = np.array(embeddings).astype("float32")
else:
    print("No persisted data found. Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents after cleaning.")

    print("Generating embeddings (this may take a few minutes)...")
    embeddings = generate_embeddings(documents)
    embeddings = np.array(embeddings).astype("float32")

    print("Persisting to disk...")
    save_to_disk(embeddings, documents)
    index = None  # Will be built by VectorSearch

# Build the vector search engine
if index is not None:
    vector_search = VectorSearch(embeddings, documents, index=index)
else:
    vector_search = VectorSearch(embeddings, documents)

# ---------------------------------------------------------------------------
# Fuzzy Clustering
# ---------------------------------------------------------------------------
# BIC evaluation was run previously and determined optimal k=8.
# BIC scores: k=8: -24267231 (lowest/best), k=9: -23723112, ..., k=21: highest
# We hardcode k=8 to avoid re-running BIC on every startup (~10 min).
# To re-evaluate, uncomment the lines below:
# bic_results, optimal_k = FuzzyCluster.evaluate_cluster_count(
#     embeddings, k_range=range(8, 22)
# )
optimal_k = 8

print(f"\nTraining fuzzy cluster model with k={optimal_k}...")
cluster_model = FuzzyCluster(n_clusters=optimal_k)
cluster_model.train(embeddings)

# Print cluster analysis to logs for evidence of meaningful clusters
print("\nCluster analysis:")
analysis = cluster_model.analyze_clusters(embeddings, documents, n_examples=2)
for c in analysis["clusters"]:
    print(f"  Cluster {c['cluster_id']}: {c['size']} documents")
print(f"\nBoundary cases (high entropy — uncertain membership):")
for bc in analysis["boundary_cases"]:
    print(f"  Entropy={bc['entropy']:.3f}, top memberships: {bc['top_memberships']}")
    print(f"    Snippet: {bc['snippet'][:100]}...")

# ---------------------------------------------------------------------------
# Semantic Cache: Cluster-accelerated, threshold-tunable
# ---------------------------------------------------------------------------

cache = SemanticCache(cluster_model=cluster_model, threshold=0.85)


# ---------------------------------------------------------------------------
# API Models & Endpoints
# ---------------------------------------------------------------------------

class Query(BaseModel):
    query: str


@app.post("/query")
def query_endpoint(q: Query):
    """
    Semantic search endpoint.

    1. Encode the query into an embedding.
    2. Check the semantic cache for a similar previous query.
    3. On HIT: return cached result with matched_query and similarity_score.
    4. On MISS: perform vector search + clustering, cache the result, return it.

    Response schema matches the specification exactly.
    """
    query_embedding = model.encode([q.query])[0]

    # --- Cache lookup ---
    hit, entry, sim = cache.lookup(query_embedding, q.query)

    if hit:
        return {
            "query": q.query,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": float(sim),
            "result": entry["result"]["result"],
            "dominant_cluster": entry["result"].get("dominant_cluster"),
        }

    # --- Cache miss: compute fresh result ---
    results = vector_search.search(np.array([query_embedding]).astype("float32"))

    dominant_cluster, cluster_probs = cluster_model.predict(
        np.array([query_embedding])
    )

    response = {
        "query": q.query,
        "cache_hit": False,
        "result": results,
        "dominant_cluster": int(dominant_cluster),
        "cluster_distribution": {
            int(i): round(float(p), 4)
            for i, p in enumerate(cluster_probs)
            if p > 0.01  # Only include clusters with >1% membership for readability
        },
    }

    # Store in cache for future similar queries
    cache.store(query_embedding, q.query, response, dominant_cluster)

    return response


@app.get("/cache/stats")
def stats_endpoint():
    """
    Return current cache statistics.

    Response keys match the specification:
    - total_entries, hit_count, miss_count, hit_rate
    """
    return cache.stats()


@app.delete("/cache")
def clear_endpoint():
    """Flush the cache entirely and reset all stats."""
    cache.clear()
    return {"message": "cache cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)