"""
Vector Search Module — NCERT Science Textbooks

Design Decision:
- FAISS IndexFlatL2 performs exact (brute-force) L2 nearest-neighbor search.
- For NCERT science textbook passages, exact search is fast enough and avoids
  the complexity of approximate methods.
- Results include metadata (class, chapter, subject) and a normalized
  similarity score (0–100%) derived from L2 distance.
"""

import re
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple, Set


class VectorSearch:

    def __init__(self, embeddings, documents, index=None):
        """
        Initialize the vector search engine.

        Parameters:
            embeddings: np.ndarray of shape (n_docs, dim) — document embeddings
            documents: list of dict — structured documents with metadata
            index: optional pre-built FAISS index (used when loading from disk)
        """
        self.documents = documents
        self.embeddings = embeddings

        if index is not None:
            self.index = index
        else:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(np.array(embeddings).astype("float32"))

    def _extract_keywords(self, passage: str, query: str, n: int = 5) -> List[str]:
        """
        Extract keywords from the passage that are relevant to the query.
        Uses simple word overlap between query terms and passage words.
        """
        query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
        passage_words = re.findall(r'\b[a-zA-Z]{3,}\b', passage.lower())

        # Stopwords to skip
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
            'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
            'have', 'been', 'this', 'that', 'with', 'they', 'from',
            'were', 'which', 'their', 'will', 'what', 'when', 'where',
            'there', 'about', 'also', 'these', 'those', 'into', 'than',
            'some', 'such', 'each', 'other', 'more', 'very', 'does',
        }

        # Find matching keywords (query terms found in passage)
        matched_keywords = []
        seen = set()
        for word in passage_words:
            if word in query_words and word not in stopwords and word not in seen:
                matched_keywords.append(word)
                seen.add(word)

        # If we don't have enough matches, add frequent passage terms
        if len(matched_keywords) < n:
            word_freq = {}
            for word in passage_words:
                if word not in stopwords and word not in seen:
                    word_freq[word] = word_freq.get(word, 0) + 1

            sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
            for word, freq in sorted_words:
                if len(matched_keywords) >= n:
                    break
                matched_keywords.append(word)
                seen.add(word)

        final_keywords = []
        for i in range(min(n, len(matched_keywords))):
            final_keywords.append(matched_keywords[i])
        return final_keywords

    def _format_title(self, title: str) -> str:
        """
        Format the chapter title according to pedagogical standards:
        - Strip "Chapter X:" prefixes.
        - Capitalize important words.
        - Lowercase common articles/prepositions (except if first word).
        - Specifically ensuring 'Cell: the Unit of Life' style.
        """
        if not title:
            return "Unknown Chapter"
            
        # 1. Remove "Chapter X:" or similar prefixes
        # Handles "Chapter 8:", "Chapter 8 - ", "CHAPTER 8 ", etc.
        clean_title = re.sub(r'(?i)^chapter\s+\d+\s*[:\-.]*\s*', '', title).strip()
        
        if not clean_title or clean_title.isdigit():
            return f"Chapter {title}" if title.isdigit() else title

        # Words to keep lowercase
        lowercase_words = {
            'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 
            'to', 'from', 'by', 'in', 'of', 'as', 'if'
        }
        
        # Split by spaces and handle capitalization
        words = clean_title.lower().split()
        formatted_words = []
        
        for i, word in enumerate(words):
            # Clean punctuation from word for lookup (e.g. "life." -> "life")
            clean_word = re.sub(r'[^a-z]', '', word)
            
            is_first = (i == 0)
            # Check if previous word ended with a colon
            is_after_colon = (i > 0 and formatted_words[-1].endswith(':'))
            
            # Special case for user: "Cell: the Unit of Life"
            # In standard title case, "the" after a colon IS capitalized.
            # But the user specifically asked for "the" (lowercase).
            
            if is_first:
                formatted_words.append(word.capitalize())
            elif clean_word in lowercase_words:
                formatted_words.append(word.lower())
            else:
                formatted_words.append(word.capitalize())
        
        return " ".join(formatted_words)

    def _l2_to_score(self, distances):
        """
        Convert L2 distances to similarity scores (0–100%).
        Uses exponential decay: score = exp(-distance / scale)
        """
        # Normalize — smaller L2 = more similar
        # Use a scaling factor based on the embedding dimension
        scale = 2.0  # Tuned for 384-dim embeddings
        scores = np.exp(-distances / scale) * 100
        return scores

    def search(self, query_embedding, query_text="", k=6):
        """
        Find the k most similar documents to the query embedding.

        Returns a list of dicts matching the API specification:
        { class, chapter, passage, score, subject, keywords }
        """
        query = np.array(query_embedding).astype("float32")
        D, I = self.index.search(query, k)

        results = []
        scores = self._l2_to_score(D[0])

        for rank, (idx, dist, score) in enumerate(zip(I[0], D[0], scores)):
            if idx < 0 or idx >= len(self.documents):
                continue

            doc = self.documents[idx]
            passage = doc.get("passage", doc) if isinstance(doc, dict) else str(doc)
            keywords = self._extract_keywords(passage, query_text)
            
            chapter = doc.get("chapter", "Unknown") if isinstance(doc, dict) else "Unknown"
            formatted_chapter = self._format_title(chapter)

            result = {
                "class": doc.get("class", 0) if isinstance(doc, dict) else 0,
                "chapter": formatted_chapter,
                "passage": passage,
                "score": round(float(score), 2),
                "subject": doc.get("subject", "Science") if isinstance(doc, dict) else "Science",
                "keywords": keywords,
                "page": doc.get("page", 1) if isinstance(doc, dict) else 1,
                "file": doc.get("file", "") if isinstance(doc, dict) else "",
            }
            results.append(result)

        return results