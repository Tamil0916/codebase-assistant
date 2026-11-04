import re

from rank_bm25 import BM25Okapi

from vectorstore import search as vector_search


STOP = {"the", "is", "are", "does", "do", "how", "where", "what", "which", "when",
        "why", "in", "of", "to", "for", "and", "or", "an", "it", "this", "that",
        "with", "can", "flask", "use", "used", "get", "gets", "from", "by", "on"}


def tokenize(text):
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return [t for t in re.split(r"[^A-Za-z0-9]+", text.lower())
            if len(t) > 1 and t not in STOP]


class HybridSearcher:
    def __init__(self, index, chunks):
        self.index = index
        self.chunks = chunks
        docs = []
        for c in chunks:
            m = c["metadata"]
            # repeat the symbol name so exact-name matches weigh more
            name = (m["symbol_name"] + " ") * 3
            docs.append(tokenize(f'{m["file_path"]} {name} {c["text"]}'))
        self.bm25 = BM25Okapi(docs)

    def search(self, query, k=6, pool=40, rrf_k=60):
        # 1. vector ranking
        vec_hits = vector_search(query, self.index, self.chunks, k=pool)
        vec_ids = [self.chunks.index(c) for _, c in vec_hits]

        # 2. keyword ranking
        scores = self.bm25.get_scores(tokenize(query))
        bm_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:pool]

        # 3. fuse with reciprocal rank fusion
        fused = {}
        for ranking in (vec_ids, bm_ids):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0) + 1 / (rrf_k + rank + 1)

        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [(fused[i], self.chunks[i]) for i in top]