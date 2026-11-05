import os

from dotenv import load_dotenv
from groq import Groq

from hybrid import HybridSearcher
from rerank import rerank
from vectorstore import load_index
from rerank import rerank, rerank_fused

load_dotenv()

# Model names change over time. Set GROQ_MODEL in .env
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

BASE_RULES = """You are a senior software engineer helping a developer understand a codebase.
Rules:
- Answer ONLY using the code snippets provided in the context.
- Cite every claim with its source in the form [file_path:start_line-end_line].
- If the context does not contain the answer, say "I couldn't find that in the indexed code." Do not guess.
- Be concise and use short code excerpts only when they help."""

MODES = {
    "ask": "Answer the developer's question about the codebase.",
    "summarize": "Summarize what the relevant code does: its purpose, its key functions or classes, and how they fit together.",
    "debug": "The developer pasted an error or a problem description. Using the code in the context, explain the likely cause and suggest a fix. Quote the exact error message from the code if it appears in the context. If the code that raises this error is NOT in the context, say so plainly instead of attributing it to other code. Say clearly when you are unsure.",
}


def is_test_file(path):
    p = path.lower().replace("\\", "/")
    return "/tests/" in f"/{p}" or "/test_" in f"/{p}" or p.endswith("_test.py")

def expand_query(query):
    """Turn a vague question into identifiers/keywords likely to appear in source code."""
    messages = [
        {"role": "system", "content":
            "Convert the developer's question about a codebase into 8-12 keywords "
            "likely to appear in the source code that answers it: function names, "
            "class names, decorators, identifiers. Output only the keywords, "
            "separated by spaces. No explanation."},
        {"role": "user", "content": query},
    ]
    for kwargs in ({"reasoning_effort": "low"}, {}):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, temperature=0,
                max_tokens=800, **kwargs)
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as e:
            print("expand_query failed:", type(e).__name__, e)
    return ""


def retrieve(query, searcher, k=6, include_tests=False, use_rerank=True, extra_terms=""):
    pool = searcher.search(f"{query} {extra_terms}".strip(), k=40)
    if not include_tests:
        pool = [h for h in pool
                if not is_test_file(h[1]["metadata"]["file_path"])
                and not h[1]["metadata"]["file_path"].endswith(".md")]
    if not pool:
        return []
    if use_rerank:
        return rerank_fused(query, pool[:30], k=k)
    return pool[:k]


def format_context(hits):
    parts = []
    for score, c in hits:
        m = c["metadata"]
        header = f'[{m["file_path"]}:{m["start_line"]}-{m["end_line"]}] {m["symbol_name"]}'
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def answer(query, searcher, mode="ask", k=6, include_tests=False):
    extra = expand_query(query) if mode == "ask" else ""
    hits = retrieve(query, searcher,
                    k=8 if mode == "summarize" else k,
                    include_tests=include_tests,
                    use_rerank=(mode != "summarize"),
                    extra_terms=extra)
    if not hits:
        return "I couldn't find that in the indexed code.", []

    messages = [
        {"role": "system", "content": f"{BASE_RULES}\n\nTask: {MODES[mode]}"},
        {"role": "user", "content": f"Context:\n{format_context(hits)}\n\nQuestion:\n{query}"},
    ]
    resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.1)
    return resp.choices[0].message.content, hits


if __name__ == "__main__":
    index, chunks = load_index()
    searcher = HybridSearcher(index, chunks)

    tests = [
        ("ask", "Where is the request context created and what does it do?"),
        ("summarize", "the Flask routing system"),
        ("debug", "RuntimeError: Working outside of application context"),
        ("ask", "How does Flask implement WebSocket support?"),
    ]
    for mode, q in tests:
        print("=" * 70)
        print(f"[{mode}] {q}\n")
        text, hits = answer(q, searcher, mode=mode)
        print(text)