import os
import re
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import rag
from chunker import chunk_repo
from hybrid import HybridSearcher
from ingest import clone_repo, load_code_files
from vectorstore import build_index, load_index

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPOS_DIR = os.path.join(ROOT, "repos")
INDEX_DIR = os.path.join(ROOT, "index")
MAX_CHUNKS = 5000  # keep indexing fast on a laptop

app = FastAPI(title="Codebase Assistant API")
_searchers = {}  # repo_id -> HybridSearcher (in-memory cache)

GITHUB_RE = re.compile(r"^https://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")


def repo_id_from_url(url: str) -> str:
    m = GITHUB_RE.match(url.strip())
    if not m:
        raise HTTPException(400, "Only public GitHub URLs like https://github.com/owner/repo are supported.")
    return f"{m.group(1)}__{m.group(2)}"


def get_searcher(repo_id: str) -> HybridSearcher:
    if repo_id in _searchers:
        return _searchers[repo_id]
    index_dir = os.path.join(INDEX_DIR, repo_id)
    if not os.path.exists(os.path.join(index_dir, "chunks.faiss")):
        raise HTTPException(404, f"Repo '{repo_id}' is not indexed. Call POST /index first.")
    index, chunks = load_index(index_dir)
    _searchers[repo_id] = HybridSearcher(index, chunks)
    return _searchers[repo_id]


class IndexRequest(BaseModel):
    repo_url: str
    force: bool = False


class AskRequest(BaseModel):
    repo_id: str
    question: str
    mode: Literal["ask", "summarize", "debug"] = "ask"
    include_tests: bool = False


@app.get("/repos")
def list_repos():
    if not os.path.isdir(INDEX_DIR):
        return {"repos": []}
    ids = [d for d in os.listdir(INDEX_DIR)
           if os.path.exists(os.path.join(INDEX_DIR, d, "chunks.faiss"))]
    return {"repos": sorted(ids)}


@app.post("/index")
def index_repo(req: IndexRequest):
    repo_id = repo_id_from_url(req.repo_url)
    index_dir = os.path.join(INDEX_DIR, repo_id)
    already = os.path.exists(os.path.join(index_dir, "chunks.faiss"))
    if already and not req.force:
        return {"repo_id": repo_id, "status": "already indexed"}

    try:
        root = clone_repo(req.repo_url, os.path.join(REPOS_DIR, repo_id))
    except Exception as e:
        raise HTTPException(400, f"Could not clone repo: {e}")

    files = load_code_files(root)
    print("DEBUG root:", root, "| files found:", len(files))
    chunks = chunk_repo(files, root)
    if not chunks:
        raise HTTPException(400, "No supported code files found in this repo.")
    if len(chunks) > MAX_CHUNKS:
        raise HTTPException(400, f"Repo too large ({len(chunks)} chunks, limit {MAX_CHUNKS}).")

    build_index(chunks, index_dir)
    _searchers.pop(repo_id, None)
    return {"repo_id": repo_id, "status": "indexed", "files": len(files), "chunks": len(chunks)}


@app.post("/ask")
def ask(req: AskRequest):
    searcher = get_searcher(req.repo_id)
    try:
        text, hits = rag.answer(req.question, searcher, mode=req.mode,
                                include_tests=req.include_tests)
    except Exception as e:
        raise HTTPException(502, f"LLM request failed: {type(e).__name__}")

    sources = []
    for score, c in hits:
        m = c["metadata"]
        sources.append({
            "file_path": m["file_path"],
            "symbol": m["symbol_name"],
            "start_line": m["start_line"],
            "end_line": m["end_line"],
            "score": round(score, 3),
        })
    return {"answer": text, "sources": sources}