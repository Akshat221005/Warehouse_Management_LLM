import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from docx import Document

# IMPORTANT: new LangChain import (for modern versions)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------
# Paths (absolute, safe)
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DIR = os.path.join(BASE_DIR, "vectorstore")

# -------------------------
# Models
# -------------------------

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Utilities
# -------------------------

def load_text_from_file(path: str) -> str:
    if path.lower().endswith(".docx"):
        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    elif path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError("Unsupported file type. Use .txt or .docx only.")

# -------------------------
# Main ingestion
# -------------------------

def ingest_document(path: str):
    print("CWD:", os.getcwd())
    print(f"Ingesting: {path}")

    text = load_text_from_file(path)

    print("Sample extracted text:")
    print(text[:500])
    print("-" * 50)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    print(f"Total chunks: {len(chunks)}")

    embeddings = EMBED_MODEL.encode(chunks, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(VECTOR_DIR, exist_ok=True)

    index_path = os.path.join(VECTOR_DIR, "index.faiss")
    chunks_path = os.path.join(VECTOR_DIR, "chunks.pkl")

    faiss.write_index(index, index_path)

    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print("Ingestion complete.")
