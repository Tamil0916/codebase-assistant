from vectorstore import load_index
from hybrid import HybridSearcher
import rag

index, chunks = load_index("index/Tamil0916__codebase-assistant")
searcher = HybridSearcher(index, chunks)

q = "What endpoints does the API expose?"
extra = rag.expand_query(q)
print("EXPANSION:", repr(extra))

pool = searcher.search(f"{q} {extra}".strip(), k=40)
pool = [h for h in pool if not rag.is_test_file(h[1]["metadata"]["file_path"])
        and not h[1]["metadata"]["file_path"].endswith(".md")]

print("\nCandidate pool (before reranking):")
for i, (s, c) in enumerate(pool[:30], 1):
    m = c["metadata"]
    print(f'{i:2}  {m["file_path"]}  [{m["symbol_name"]}]')

print("\nAfter reranking (what the LLM sees):")
for s, c in rag.rerank(q, pool[:30], k=6):
    m = c["metadata"]
    print(f'{s:7.2f}  {m["file_path"]}  [{m["symbol_name"]}]')
    print("\nAfter FUSED reranking:")
from rerank import rerank_fused
for s, c in rerank_fused(q, pool[:30], k=6):
    m = c["metadata"]
    print(f'{s:.4f}  {m["file_path"]}  [{m["symbol_name"]}]')

print("\nAfter FUSED reranking:")
from rerank import rerank_fused
for s, c in rerank_fused(q, pool[:30], k=6):
    m = c["metadata"]
    print(f'{s:.4f}  {m["file_path"]}  [{m["symbol_name"]}]')