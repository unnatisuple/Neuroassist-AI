"""
NeuroAssist AI v2 — RAG Index Builder
Builds the FAISS vector index and document store from guideline text files.
Seeds sample guidelines for testing.
"""

import os
import sys
import json
from pathlib import Path
from loguru import logger

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import settings


SAMPLE_GUIDELINES = """
WHO Dementia Guidelines 2023 - Risk Reduction:
Physical activity is recommended for adults with normal cognition to reduce the risk of cognitive decline.
A healthy, balanced diet (such as the Mediterranean diet) may be recommended to reduce the risk of cognitive decline/dementia.
Cognitive training may be offered to older adults with normal cognition and mild cognitive impairment (MCI) to reduce the risk of cognitive decline.
There is insufficient evidence to recommend vitamins, minerals, or dietary supplements to reduce the risk of cognitive decline.

Alzheimer's Association Guidelines 2024 - Diagnosis:
Diagnosis of Alzheimer's dementia should be based on clinical history, cognitive assessments (MMSE, MoCA), and neuroimaging.
Brain MRI should be used to rule out other causes of dementia (such as vascular lesions or tumors) and to assess regional atrophy.
Symmetric or asymmetric temporal lobe atrophy, particularly in the hippocampus, is a key neuroimaging biomarker for Alzheimer's disease.
In early stages, MRI may show mild hippocampal atrophy, which progresses to generalized cerebral atrophy in moderate-to-severe stages.

WHO Dementia Guidelines 2023 - Clinical Recommendation for Mild Stage:
For patients diagnosed with mild dementia, cognitive stimulation therapy (CST) should be considered.
Regular physical exercise, structured walking, and social engagement are recommended to preserve daily living functions.
Clinicians should screen for comorbid conditions (such as depression, hypertension, and diabetes) that can accelerate cognitive decline.
Caregivers of people with dementia should be provided with structured training and psychosocial support (e.g., WHO iSupport).

NICE Guidelines NG97 - Dementia Care:
Offer occupational therapy to people living with dementia to help them maintain independence in activities of daily living.
For moderate dementia, consider non-pharmacological treatments (such as music therapy, art therapy, or reminiscence therapy) to manage behavioral symptoms.
Acetylcholinesterase (AChE) inhibitors (such as donepezil, galantamine, and rivastigmine) are recommended for mild to moderate Alzheimer's disease.
Memantine is recommended for moderate Alzheimer's disease if AChE inhibitors are not tolerated, or for severe Alzheimer's disease.
"""


def chunk_text(text: str, source_name: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """Split text into overlapping chunks."""
    chunks = []
    # Basic paragraph/line splitting
    paragraphs = text.split("\n\n")

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # If paragraph is small, keep it as is
        if len(paragraph) <= chunk_size:
            chunks.append({
                "text": paragraph,
                "source": source_name,
            })
        else:
            # Split into chunks of chunk_size with overlap
            words = paragraph.split()
            current_words = []
            current_len = 0

            for word in words:
                current_words.append(word)
                current_len += len(word) + 1
                if current_len >= chunk_size:
                    # Save chunk
                    chunks.append({
                        "text": " ".join(current_words),
                        "source": source_name,
                    })
                    # Slide window (overlap)
                    # Keep last 15 words for overlap
                    current_words = current_words[-15:]
                    current_len = sum(len(w) + 1 for w in current_words)

            if current_words:
                chunks.append({
                    "text": " ".join(current_words),
                    "source": source_name,
                })

    return chunks


def build_index():
    logger.info("Initializing RAG Index Builder...")

    # Ensure directories exist
    os.makedirs(settings.rag_docs_dir, exist_ok=True)
    os.makedirs(settings.faiss_index_dir, exist_ok=True)

    # Seed sample guidelines if directory is empty
    sample_file = os.path.join(settings.rag_docs_dir, "sample_guidelines.txt")
    if not os.listdir(settings.rag_docs_dir):
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_GUIDELINES.strip())
        logger.info(f"Seeded sample guidelines file at {sample_file}")

    # Gather documents
    documents = []
    for filename in os.listdir(settings.rag_docs_dir):
        if filename.endswith(".txt") or filename.endswith(".md"):
            file_path = os.path.join(settings.rag_docs_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            source_name = filename.rsplit(".", 1)[0].replace("_", " ").title()
            logger.info(f"Processing source file: {filename} ({source_name})")

            # Chunk text
            chunks = chunk_text(content, source_name)
            documents.extend(chunks)

    if not documents:
        logger.error("No text documents found to index.")
        return False

    logger.info(f"Total document chunks extracted: {len(documents)}")

    # Load SentenceTransformer model
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {settings.embedding_model}...")
        embedder = SentenceTransformer(settings.embedding_model)
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer: {e}")
        return False

    # Extract texts and compute embeddings
    texts = [doc["text"] for doc in documents]
    logger.info("Computing embeddings...")
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Create FAISS index
    import faiss
    dimension = embeddings.shape[1]
    logger.info(f"Creating FAISS index with dimension {dimension}...")

    # Using L2 distance index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))

    # Save index and document store
    index_path = os.path.join(settings.faiss_index_dir, "index.faiss")
    docs_path = os.path.join(settings.faiss_index_dir, "documents.json")

    try:
        faiss.write_index(index, index_path)

        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=2)

        logger.info(f"[SUCCESS] FAISS index saved to {index_path}")
        logger.info(f"[SUCCESS] Document store saved to {docs_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save RAG index artifacts: {e}")
        return False


if __name__ == "__main__":
    success = build_index()
    sys.exit(0 if success else 1)
