from sentence_transformers import CrossEncoder

_model = None


def rerank(query, hits, k=6):
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    chunks = [c for _, c in hits]
    pairs = [(query, f'{c["metadata"]["symbol_name"]}\n{c["text"][:1500]}') for c in chunks]
    scores = _model.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [(float(s), c) for s, c in ranked[:k]]

def rerank_fused(query, hits, k=6, rrf_k=60):
    """Blend the hybrid ranking (input order) with the cross-encoder ranking."""
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    chunks = [c for _, c in hits]
    pairs = [(query, f'{c["metadata"]["symbol_name"]}\n{c["text"][:1500]}') for c in chunks]
    ce_scores = _model.predict(pairs)

    ce_order = sorted(range(len(chunks)), key=lambda i: ce_scores[i], reverse=True)
    ce_rank = {i: r for r, i in enumerate(ce_order)}

    fused = []
    for i, c in enumerate(chunks):  # i is also the hybrid rank
        score = 1 / (rrf_k + i + 1) + 1 / (rrf_k + ce_rank[i] + 1)
        fused.append((score, c))
    fused.sort(key=lambda x: x[0], reverse=True)
    return fused[:k]