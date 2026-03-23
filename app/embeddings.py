"""
Embedding & Vector Database Setup — NCERT Science Textbooks

Design Decisions:
-----------------
1. Model Choice: all-MiniLM-L6-v2
   - 384-dimensional embeddings — compact for fast similarity, yet semantically rich.
   - Trained on 1B+ sentence pairs; robust for educational science text.

2. PDF Ingestion:
   - We parse every PDF from the '6-12th science dataset/' folder using PyMuPDF.
   - Filenames encode the class and subject (e.g., "Science-Class-8.pdf",
     "Physics---Part-1---Class-11.pdf"). We extract these metadata fields.
   - Text is chunked into passages of ~500 characters at paragraph boundaries so
     each embedding represents a coherent idea, not a random slice.

3. Metadata per passage:
   - class_number (int): The NCERT class (6–11)
   - subject (str): Physics / Chemistry / Biology / Science
   - chapter (str): Chapter title extracted from the text heading patterns
   - passage (str): The text chunk

4. Persistence:
   - FAISS index, raw embeddings, and structured documents are saved to disk.
"""

import os
import re
import json
import logging
import numpy as np
import faiss
import fitz  # PyMuPDF
import requests

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Model constants
MODEL_NAME = "all-MiniLM-L6-v2"
HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/{MODEL_NAME}"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "6-12th science dataset")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
DOCUMENTS_PATH = os.path.join(DATA_DIR, "documents.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")

# Global model instance (lazy loaded)
_model = None

def get_model():
    """
    Lazy-load the SentenceTransformer model.
    Falls back to a 'cloud' mode if dependencies are missing.
    """
    global _model
    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading local SentenceTransformer model: {MODEL_NAME}")
        # Use local_files_only=True to ensure we don't try to download if it might fail
        # but wait, first run needs download. Let's just catch and log properly.
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Local model loaded successfully.")
    except Exception as e:
        import traceback
        logger.warning(f"Could not load local model. Switching to HuggingFace Inference API.")
        logger.debug(traceback.format_exc())
        _model = "CLOUD_API"
    
    return _model

def encode_texts(texts):
    """
    Encode a list of texts into embeddings.
    Works locally via torch or remotely via HF Inference API.
    """
    model = get_model()
    
    if model == "CLOUD_API":
        # Remote inference fallback (ideal for Vercel/Lambda)
        # Fix: Using 'models/' instead of 'pipeline/feature-extraction/' for newer API
        API_URL = f"https://api-inference.huggingface.co/models/sentence-transformers/{MODEL_NAME}"
        headers = {}
        token = os.getenv("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
            
        try:
            response = requests.post(API_URL, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}})
            response.raise_for_status()
            return np.array(response.json())
        except Exception as e:
            logger.error(f"HF API encoding failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            raise
    else:
        # Local inference
        return model.encode(texts, show_progress_bar=False)


def _parse_filename(filename):
    """
    Extract class number and subject from the PDF filename.

    Examples:
        'Science---Class-6.pdf'     -> (6, 'Science')
        'Physics---Part-1---Class-11.pdf' -> (11, 'Physics')
        'Chemistry---Part-2---Class-11.pdf' -> (11, 'Chemistry')
        'Biology-Class-11.pdf'     -> (11, 'Biology')
        'Science-Class-10.pdf'     -> (10, 'Science')
    """
    name = filename.replace(".pdf", "").replace(".PDF", "")

    # Extract class number
    class_match = re.search(r'[Cc]lass[- ]*(\d+)', name)
    class_number = int(class_match.group(1)) if class_match else 0

    # Extract subject — everything before the first '---' or '-Class'
    subject = re.split(r'---|-[Cc]lass', name)[0].strip()
    subject = subject.replace("-", " ").strip()

    # Normalize subjects
    subject_lower = subject.lower()
    if "physics" in subject_lower:
        subject = "Physics"
    elif "chemistry" in subject_lower:
        subject = "Chemistry"
    elif "biology" in subject_lower:
        subject = "Biology"
    else:
        subject = "Science"

    return class_number, subject


def _extract_chapters_from_text(full_text):
    """
    Attempt to find chapter boundaries in the extracted PDF text.
    NCERT books typically have chapter headings like:
        'Chapter 1', 'CHAPTER 2', 'chapter 1: Title', etc.

    Returns a list of (chapter_title, chapter_text) tuples.
    """
    # Pattern to match chapter headings
    chapter_pattern = re.compile(
        r'(?:^|\n)\s*(Chapter\s+\d+[\s:.\-]*[^\n]*)',
        re.IGNORECASE
    )

    matches = list(chapter_pattern.finditer(full_text))

    if not matches:
        # If no chapter headings found, treat entire text as one chapter
        return [("Full Text", full_text)]

    chapters = []
    for i, match in enumerate(matches):
        chapter_title = match.group(1).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        chapter_text = full_text[start:end].strip()
        chapters.append((chapter_title, chapter_text))

    return chapters


def _chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into passages of approximately chunk_size characters.
    Splits at paragraph boundaries (\n\n) when possible to keep passages coherent.
    """
    # Split into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 1 <= chunk_size:
            current_chunk += (" " + para) if current_chunk else para
        else:
            if current_chunk and len(current_chunk) >= 40:
                chunks.append(current_chunk.strip())
            current_chunk = para

    # Add the last chunk
    if current_chunk and len(current_chunk) >= 40:
        chunks.append(current_chunk.strip())

    # If any chunk is still very long, force-split it
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > chunk_size * 2:
            # Split at sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            sub_chunk = ""
            for sent in sentences:
                if len(sub_chunk) + len(sent) + 1 <= chunk_size:
                    sub_chunk += (" " + sent) if sub_chunk else sent
                else:
                    if sub_chunk and len(sub_chunk) >= 40:
                        final_chunks.append(sub_chunk.strip())
                    sub_chunk = sent
            if sub_chunk and len(sub_chunk) >= 40:
                final_chunks.append(sub_chunk.strip())
        else:
            final_chunks.append(chunk)

    return final_chunks


def load_documents():
    """
    Load and process all NCERT Science PDFs.
    Returns a list of dicts with: { class, subject, chapter, passage, page, file }
    """
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    pdf_files = sorted([f for f in os.listdir(DATASET_DIR) if f.lower().endswith(".pdf")])
    all_documents = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATASET_DIR, pdf_file)
        class_number, subject = _parse_filename(pdf_file)

        print(f"  Processing: {pdf_file} (Class {class_number}, {subject})")
        current_chapter = "General Context"

        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if not text.strip():
                    continue
                
                # Improved chapter detection
                # 1. Broad search for "Chapter X" pattern
                chapter_match = re.search(r'(?:^|\n)\s*Chapter\s+(\d+)', text, re.IGNORECASE)
                
                if chapter_match:
                    header_num = chapter_match.group(1)
                    # Try to find the title - usually on the same or next line
                    # Look for everything after the number on the same line
                    post_num = text[chapter_match.end():].split('\n')[0].strip()
                    
                    # Also look at the next line if the current line is short
                    lines = text.split('\n')
                    current_line_idx = -1
                    for idx, line in enumerate(lines):
                        if f"Chapter {header_num}" in line or f"CHAPTER {header_num}" in line:
                            current_line_idx = idx
                            break
                    
                    next_line = lines[current_line_idx + 1].strip() if current_line_idx != -1 and current_line_idx + 1 < len(lines) else ""
                    
                    # Heuristic for title
                    potential_title = post_num if len(post_num) > 3 else next_line
                    
                    # More aggressive cleaning
                    # 1. Remove years like 2019-20, 2020-21, 2021
                    clean_title = re.sub(r'(?:20\d{2}[-\d]*)', '', potential_title).strip()
                    # 2. Remove common NCERT artifacts
                    clean_title = re.sub(r'(?i)rationalised|content|preface|acknowledgement', '', clean_title).strip()
                    # 3. Remove punctuation
                    clean_title = re.sub(r'^[:\-\s\.,]+|[:\-\s\.,]+$', '', clean_title).strip()
                    
                    # Final validation: title should be words, not just numbers
                    if len(clean_title) > 3 and not re.match(r'^[\d\-\s\.,]+$', clean_title):
                        # Strip any remaining "Chapter" or "CHAPTER" from the start
                        final_title = re.sub(r'(?i)chapter\s*\d*\s*[:\-]*\s*', '', clean_title).strip()
                        current_chapter = final_title if len(final_title) > 2 else f"Chapter {header_num}"
                    else:
                        current_chapter = f"Chapter {header_num}"
                        
                # Special Case: Strip "Chapter" prefix even from fallback
                current_chapter = re.sub(r'(?i)chapter\s+', '', current_chapter).strip()

                passages = _chunk_text(text)
                for passage in passages:
                    passage_clean = re.sub(r'\s+', ' ', passage).strip()
                    if len(passage_clean) < 40:
                        continue

                    all_documents.append({
                        "class": class_number,
                        "subject": subject,
                        "chapter": current_chapter,
                        "passage": passage_clean,
                        "page": page_num + 1,
                        "file": pdf_file
                    })
            doc.close()
        except Exception as e:
            print(f"  WARNING: Could not read {pdf_file}: {e}")
            continue

    return all_documents


def generate_embeddings(documents):
    """
    Generate dense vector embeddings for all document passages.
    Uses the passage text only (not metadata) for embedding.
    """
    texts = [doc["passage"] for doc in documents]
    embeddings = encode_texts(texts)
    return embeddings


def save_to_disk(embeddings, documents):
    """
    Persist the FAISS index, raw embeddings, and structured documents to disk.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Save FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save raw embeddings
    np.save(EMBEDDINGS_PATH, embeddings)

    # Save structured documents (with metadata)
    with open(DOCUMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)


def load_from_disk():
    """
    Load persisted FAISS index, embeddings, and documents from disk.
    Returns (None, None, None) if data hasn't been persisted yet.
    """
    if not all(os.path.exists(p) for p in [FAISS_INDEX_PATH, DOCUMENTS_PATH, EMBEDDINGS_PATH]):
        return None, None, None

    index = faiss.read_index(FAISS_INDEX_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    return index, embeddings, documents