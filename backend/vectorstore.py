import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = "index"

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def text_to_embed(chunk):
    m = chunk["metadata"]
    return f'File: {m["file_path"]}\nSymbol: {m["symbol_name"]}\n\n{chunk["text"]}'


def build_index(chunks, index_dir=INDEX_DIR):
    model = get_model()
    texts = [text_to_embed(c) for c in chunks]
    emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])  # inner product = cosine (vectors are normalized)
    index.add(emb)

    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "chunks.faiss"))
    with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    return index


def load_index(index_dir=INDEX_DIR):
    index = faiss.read_index(os.path.join(index_dir, "chunks.faiss"))
    with open(os.path.join(index_dir, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks


def search(query, index, chunks, k=5):
    q = get_model().encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(q, dtype="float32"), k)
    return [(float(s), chunks[i]) for s, i in zip(scores[0], ids[0]) if i != -1]


if __name__ == "__main__":
    from ingest import clone_repo, load_code_files
    from chunker import chunk_repo

    if not os.path.exists(os.path.join(INDEX_DIR, "chunks.faiss")):
        root = clone_repo("https://github.com/pallets/flask")
        chunks = chunk_repo(load_code_files(root), root)
        print(f"Embedding {len(chunks)} chunks...")
        build_index(chunks)

    index, chunks = load_index()

    queries = [
        "How does Flask handle URL routing?",
        "Where is the request context created?",
        "How are errors and exceptions handled?",
    ]
    for q in queries:
        print("=" * 60)
        print("Q:", q)
        for score, c in search(q, index, chunks, k=3):
            m = c["metadata"]
            print(f'  {score:.3f}  {m["file_path"]} [{m["symbol_name"]}] lines {m["start_line"]}-{m["end_line"]}')