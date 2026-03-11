"""
Vector Search Module

Design Decision:
- FAISS IndexFlatL2 performs exact (brute-force) L2 nearest-neighbor search.
- For ~18,000 documents with 384-dimensional embeddings, exact search is fast
  enough (sub-millisecond) and avoids the complexity and recall trade-offs of
  approximate methods (IVF, HNSW).
- If the corpus grew to millions of documents, we would switch to an approximate
  index (e.g., IndexIVFFlat) for sub-linear search time.
"""

import faiss
import numpy as np


class VectorSearch:

    def __init__(self, embeddings, documents, index=None):
        """
        Initialize the vector search engine.

        Parameters:
            embeddings: np.ndarray of shape (n_docs, dim) — document embeddings
            documents: list of str — original document texts, aligned with embeddings
            index: optional pre-built FAISS index (used when loading from disk)
        """
        self.documents = documents
        self.embeddings = embeddings

        if index is not None:
            # Reuse a pre-built index loaded from disk to avoid redundant indexing
            self.index = index
        else:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(np.array(embeddings).astype("float32"))

    def search(self, query_embedding, k=5):
        """
        Find the k most similar documents to the query embedding.

        Returns a list of dicts with document text and L2 distance for transparency.
        """
        query = np.array(query_embedding).astype("float32")
        D, I = self.index.search(query, k)

        results = []
        for rank, (idx, dist) in enumerate(zip(I[0], D[0])):
            results.append({
                "rank": rank + 1,
                "document": self.documents[idx],
                "l2_distance": float(dist),
            })

        return results