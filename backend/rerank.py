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