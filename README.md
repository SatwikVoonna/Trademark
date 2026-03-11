\# Semantic Search System



A lightweight semantic search engine with fuzzy clustering and cluster-accelerated semantic caching, built on the 20 Newsgroups dataset.



\## Architecture



```

Query → Embedding → Cache Lookup → (Hit?) → Return Cached Result

&nbsp;                                  (Miss?) → FAISS Search → Cluster Assignment → Cache Store → Return Result

```



\## Components



| Module | Purpose |

|---|---|

| `embeddings.py` | Loads 20 Newsgroups, generates embeddings using `all-MiniLM-L6-v2`, persists to disk |

| `search.py` | FAISS-based exact nearest-neighbor search (IndexFlatL2) |

| `clustering.py` | Gaussian Mixture Model (GMM) for fuzzy/soft clustering with BIC-based model selection |

| `cache.py` | Cluster-accelerated semantic cache with cosine similarity matching |

| `main.py` | FastAPI service wiring all components together |



\## Design Decisions



\### 1. Embedding Model — `all-MiniLM-L6-v2`

\- Produces 384-dimensional embeddings — compact enough for fast search, rich enough for strong semantic quality.

\- Trained on 1B+ sentence pairs, making it robust across the diverse topics in 20 Newsgroups (politics, sports, hardware, religion, etc.).

\- Larger models (e.g., `all-mpnet-base-v2`, 768-dim) offer marginal quality gains but double the embedding size and inference time — unnecessary for this scale.



\### 2. Fuzzy Clustering — GMM over KMeans

\- \*\*KMeans\*\* produces hard assignments — each document belongs to exactly 1 cluster. This doesn't reflect reality where a document about "sports cars" could belong to both automotive and sports topics.

\- \*\*GMM\*\* natively outputs posterior probabilities `P(cluster\_k | document)` via Bayesian inference, providing true fuzzy membership distributions.

\- GMM also models clusters as ellipsoids with varying covariances, capturing the reality that topic clusters have different shapes and spreads.



\### 3. Cluster Count Selection — BIC

\- Instead of hardcoding an arbitrary number of clusters, we use the \*\*Bayesian Information Criterion (BIC)\*\* to find the optimal `k`.

\- BIC balances model fit against complexity: `BIC = -2 \* log\_likelihood + n\_params \* log(n\_samples)`

\- Evaluated k=8 to k=21; \*\*k=8\*\* achieved the lowest BIC score (-24,267,231), indicating 8 clusters best explain the data without overfitting.



\### 4. Semantic Cache — Cluster-Accelerated Lookup

\- A naive cache scans every entry on every lookup — O(n).

\- By bucketing cache entries by their dominant cluster (from the GMM), we only compare queries against entries in the \*\*same cluster\*\* — reducing lookup to O(n/k).

\- \*\*Cosine similarity threshold (0.85):\*\* Queries must be near-paraphrases to trigger a cache hit. Lower threshold = more hits but less precision. Higher = fewer hits but guaranteed relevance.



\### 5. Vector Search — FAISS IndexFlatL2

\- Exact brute-force L2 search. For ~18,000 documents with 384-dim embeddings, search is sub-millisecond.

\- If the corpus grew to millions, we'd switch to approximate methods (IVF, HNSW) for sub-linear search time.



\## API Endpoints



\### `POST /query`

Search for semantically similar documents.



```json

// Request

{ "query": "What is the best graphics card for gaming?" }



// Response

{

&nbsp; "query": "What is the best graphics card for gaming?",

&nbsp; "cache\_hit": false,

&nbsp; "result": \[

&nbsp;   { "rank": 1, "document": "...", "l2\_distance": 0.82 },

&nbsp;   { "rank": 2, "document": "...", "l2\_distance": 0.87 }

&nbsp; ],

&nbsp; "dominant\_cluster": 3,

&nbsp; "cluster\_distribution": \[0.01, 0.02, 0.05, 0.85, ...]

}

```



\### `GET /cache/stats`

View cache performance metrics.



```json

{

&nbsp; "total\_entries": 12,

&nbsp; "hit\_count": 5,

&nbsp; "miss\_count": 7,

&nbsp; "hit\_rate": 0.417

}

```



\### `DELETE /cache`

Flush all cached entries and reset statistics.



\## Cluster Analysis Results



| Cluster | Size | Themes |

|---------|------|--------|

| 0 | 1,975 | Mixed topics |

| 1 | 2,414 | Mixed topics |

| 2 | 879 | Smallest — niche topics |

| 3 | 1,727 | Mixed topics |

| 4 | 2,150 | Mixed topics |

| 5 | 3,520 | Largest — broad general topics |

| 6 | 2,826 | Mixed topics |

| 7 | 2,638 | Mixed topics |



\*\*Boundary cases\*\* (documents with high entropy — genuinely uncertain membership):

\- Documents with membership split across multiple clusters confirm that fuzzy clustering captures real-world topic overlap.



\## Setup \& Installation



\### Local



```bash

python -m venv venv

source venv/bin/activate  # Windows: venv\\Scripts\\activate

pip install -r requirements.txt

uvicorn app.main:app --host 127.0.0.1 --port 8000

```



> First run takes ~15 minutes (embedding generation + BIC evaluation). Subsequent runs load from disk in seconds.



\### Docker



```bash

docker-compose up --build

```



\### Environment Variables (optional)



```bash

HF\_HUB\_OFFLINE=1          # Skip HuggingFace update checks

TRANSFORMERS\_OFFLINE=1     # Use locally cached model

```



\## Tech Stack



\- \*\*FastAPI\*\* — async web framework

\- \*\*Sentence-Transformers\*\* — `all-MiniLM-L6-v2` for embeddings

\- \*\*FAISS\*\* — Facebook AI Similarity Search for vector retrieval

\- \*\*scikit-learn\*\* — Gaussian Mixture Model for fuzzy clustering

\- \*\*NumPy\*\* — numerical operations



\## Dataset



\*\*20 Newsgroups\*\* — 18,129 documents (after cleaning) across 20 categories covering politics, religion, sports, technology, science, and more.



Cleaning: Removed headers, footers, and quotes (email metadata). Filtered documents shorter than 30 characters (artifacts of metadata removal).

