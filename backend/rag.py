import os

from dotenv import load_dotenv
from groq import Groq

from vectorstore import load_index, search
from hybrid import HybridSearcher

load_dotenv()

# Model names change over time. If this one is rejected, check the current list at console.groq.com
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
    "debug": "The developer pasted an error or a problem description. Using the code in the context, explain the likely cause and suggest a fix. Say clearly when you are unsure.",
}


def is_test_file(path):
    p = path.lower().replace("\\", "/")
    return "/tests/" in f"/{p}" or "/test_" in f"/{p}" or p.endswith("_test.py")


_searcher = None


def retrieve(query, index, chunks, k=6, include_tests=False):
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher(index, chunks)
    hits = _searcher.search(query, k=k * 4)
    if not include_tests:
        hits = [h for h in hits
                if not is_test_file(h[1]["metadata"]["file_path"])
                and not h[1]["metadata"]["file_path"].endswith(".md")]
    return hits[:k]


def format_context(hits):
    parts = []
    for score, c in hits:
        m = c["metadata"]
        header = f'[{m["file_path"]}:{m["start_line"]}-{m["end_line"]}] {m["symbol_name"]}'
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def answer(query, index, chunks, mode="ask", k=6, include_tests=False):
    hits = retrieve(query, index, chunks, k=k, include_tests=include_tests)
    context = format_context(hits)

    messages = [
        {"role": "system", "content": f"{BASE_RULES}\n\nTask: {MODES[mode]}"},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"},
    ]
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.1
    )
    return resp.choices[0].message.content, hits


if __name__ == "__main__":
    index, chunks = load_index()

    tests = [
        ("ask", "Where is the request context created and what does it do?"),
        ("summarize", "the Flask routing system"),
        ("debug", "RuntimeError: Working outside of application context"),
        ("ask", "How does Flask implement WebSocket support?"),  # not in Flask: should say it can't find it
    ]
    for mode, q in tests:
        print("=" * 70)
        print(f"[{mode}] {q}\n")
        text, hits = answer(q, index, chunks, mode=mode)
        print(text)
        print("\nSources retrieved:")
        for s, c in hits:
            m = c["metadata"]
            print(f'  {s:.3f} {m["file_path"]} [{m["symbol_name"]}] {m["start_line"]}-{m["end_line"]}')