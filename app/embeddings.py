"""
Embedding & Vector Database Setup (Part 1)

Design Decisions:
-----------------
1. Model Choice: all-MiniLM-L6-v2
   - This model produces 384-dimensional embeddings, which is compact enough for
     fast similarity search while retaining strong semantic quality.
   - It is trained on over 1 billion sentence pairs, making it robust for
     general-purpose semantic similarity — ideal for the diverse topics in the
     20 Newsgroups dataset (politics, sports, hardware, religion, etc.).
   - Larger models (e.g., all-mpnet-base-v2) offer marginal quality gains but
     double the embedding dimension (768) and inference time, which is unnecessary
     for a lightweight search system.

2. Data Cleaning:
   - We remove headers, footers, and quotes via fetch_20newsgroups built-in flags.
     These contain metadata (email addresses, "Re:" chains, signatures) that add
     noise and would cause the model to cluster by author/thread rather than topic.
   - We additionally filter out:
     a) Empty or whitespace-only documents (no semantic content).
     b) Documents shorter than 30 characters — these are typically residual
        artifacts after header/footer removal (e.g., single words, "yes", "me too").
   - We do NOT apply aggressive lowercasing or stopword removal because the
     sentence-transformer model handles raw text natively and benefits from
     natural casing and word context.

3. Persistence:
   - FAISS index and metadata are saved to disk so embeddings are not regenerated
     on every server restart. This is critical for a production-ready service.
"""

import os
import json
import numpy as np
import faiss
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer

# Compact, high-quality model for semantic similarity (see justification above)
model = SentenceTransformer("all-MiniLM-L6-v2")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
DOCUMENTS_PATH = os.path.join(DATA_DIR, "documents.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")


def load_documents():
    """
    Load and clean the 20 Newsgroups dataset.

    Cleaning strategy:
    - Remove headers/footers/quotes to strip email metadata and reply chains.
    - Filter empty documents and very short documents (< 30 chars) that carry
      no meaningful semantic content after metadata removal.
    """
    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
    )

    cleaned = []
    for doc in dataset.data:
        text = doc.strip()
        # Skip empty or near-empty documents that are artifacts of header/footer removal
        if len(text) < 30:
            continue
        cleaned.append(text)

    return cleaned


def generate_embeddings(documents):
    """
    Generate dense vector embeddings for all documents using the sentence-transformer model.
    show_progress_bar=True provides visibility during the ~2 minute encoding process.
    """
    embeddings = model.encode(documents, show_progress_bar=True)
    return embeddings


def save_to_disk(embeddings, documents):
    """
    Persist the FAISS index, raw embeddings, and document texts to disk.
    This avoids re-encoding ~18,000 documents on every server restart.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Save FAISS index for fast retrieval
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save raw embeddings (needed for clustering and cache operations)
    np.save(EMBEDDINGS_PATH, embeddings)

    # Save document texts (needed to return search results)
    with open(DOCUMENTS_PATH, "w") as f:
        json.dump(documents, f)


def load_from_disk():
    """
    Load persisted FAISS index, embeddings, and documents from disk.
    Returns None if data has not been persisted yet.
    """
    if not all(os.path.exists(p) for p in [FAISS_INDEX_PATH, DOCUMENTS_PATH, EMBEDDINGS_PATH]):
        return None, None, None

    index = faiss.read_index(FAISS_INDEX_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    with open(DOCUMENTS_PATH, "r") as f:
        documents = json.load(f)

    return index, embeddings, documents