"""
Semantic Cache Module (Part 3)

Design Decisions:
-----------------
1. Why cluster-accelerated lookup:
   - A naive cache scans every entry on every lookup — O(n) per query.
   - By organizing cache entries into cluster buckets (using the fuzzy clustering
     from Part 2), we only compare the query against entries in the SAME cluster.
   - This reduces lookup time proportionally to the number of clusters. With 15
     clusters and uniform distribution, we scan ~1/15th of the cache.
   - As the cache grows large, this becomes critical for keeping response times low.

2. Similarity threshold (tunable parameter):
   - The threshold controls the trade-off between cache hit rate and result accuracy.
   - LOW threshold (e.g., 0.75): More cache hits, but may return results for queries
     that are only loosely related. Good for FAQ-style systems where approximate
     answers are acceptable.
   - HIGH threshold (e.g., 0.95): Fewer cache hits, but near-guarantee that cached
     results are relevant. Appropriate when precision matters more than speed.
   - DEFAULT (0.85): A balanced choice — empirically, cosine similarity > 0.85
     between MiniLM embeddings indicates paraphrases or near-identical intent.
   - The threshold is exposed as a constructor parameter so it can be tuned and
     its effect on hit_rate observed via the /cache/stats endpoint.

3. No external libraries:
   - The cache is a pure Python dict-of-lists structure. No Redis, Memcached, or
     any caching middleware. Every line of cache logic is hand-written.

4. Storing original query text:
   - Each cache entry stores the original query string so we can return
     "matched_query" on a hit, showing the user which previous query was similar.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:

    def __init__(self, cluster_model, threshold=0.85):
        """
        Initialize a cluster-accelerated semantic cache.

        Parameters:
            cluster_model: A FuzzyCluster instance used to assign queries to clusters.
                           Cache entries are bucketed by dominant cluster for efficient lookup.
            threshold: Cosine similarity threshold for cache hits.
                       Values closer to 1.0 require near-exact semantic matches.
                       Values closer to 0.0 accept looser matches.
        """
        # Cache is organized as {cluster_id: [list of entries]}
        # This leverages the clustering from Part 2 for efficient lookup
        self.cache = {}
        self.cluster_model = cluster_model
        self.threshold = threshold
        self.hit_count = 0
        self.miss_count = 0

    def _get_cluster(self, embedding):
        """Determine the dominant cluster for an embedding."""
        dominant, _ = self.cluster_model.predict(np.array([embedding]))
        return dominant

    def lookup(self, embedding, query_text, filters=None):
        """
        Search the cache for a semantically similar previous query.

        Strategy:
        1. Determine the query's dominant cluster.
        2. Only compare against cache entries in that cluster (not the entire cache).
        3. Return the best match above the similarity threshold.

        This cluster-scoped search is what makes the cache scale — instead of
        comparing against all N entries, we compare against ~N/k entries where
        k is the number of clusters.

        Returns:
            (hit, entry, similarity) where:
            - hit: bool — whether a cache hit was found
            - entry: dict or None — the matched cache entry
            - similarity: float or None — cosine similarity of the match
        """
        cluster_id = self._get_cluster(embedding)
        bucket = self.cache.get(cluster_id, [])

        best_match = None
        best_sim = -1

        for entry in bucket:
            entry_filters = entry.get("filters", {})
            req_filters = filters or {}
            if entry_filters != req_filters:
                continue

            sim = cosine_similarity(
                [embedding],
                [entry["embedding"]]
            )[0][0]

            if sim > self.threshold and sim > best_sim:
                best_sim = sim
                best_match = entry

        if best_match is not None:
            self.hit_count += 1
            return True, best_match, float(best_sim)

        self.miss_count += 1
        return False, None, None

    def store(self, embedding, query_text, result, cluster_id, filters=None):
        """
        Store a new entry in the cache, bucketed by cluster.

        Parameters:
            embedding: The query embedding vector
            query_text: The original query string (needed for matched_query response)
            result: The computed result to cache
            cluster_id: The dominant cluster for this query
        """
        if cluster_id not in self.cache:
            self.cache[cluster_id] = []

        self.cache[cluster_id].append({
            "embedding": np.array(embedding).tolist(),
            "query": query_text,
            "result": result,
            "filters": filters or {},
        })

    def stats(self):
        """
        Return cache statistics matching the required API response format.

        Keys match the specification exactly:
        - total_entries: number of cached query-result pairs
        - hit_count: number of cache hits
        - miss_count: number of cache misses
        - hit_rate: ratio of hits to total lookups
        """
        total_entries = sum(len(bucket) for bucket in self.cache.values())
        total_lookups = self.hit_count + self.miss_count

        return {
            "total_entries": total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(self.hit_count / total_lookups, 3) if total_lookups else 0,
        }

    def clear(self):
        """Flush all cache entries and reset all statistics."""
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0