from vectorstore import load_index, search
from hybrid import HybridSearcher
from rag import is_test_file
from rerank import rerank

# (question, acceptable symbol names)
TESTS = [
    ("Where is the request context created?", ["Flask.request_context"]),
    ("How does Flask generate URLs for an endpoint?", ["Flask.url_for"]),
    ("How do I register a URL route with a decorator?", ["Scaffold.route"]),
    ("How does Flask handle unhandled exceptions?", ["Flask.handle_exception"]),
    ("How are custom error handlers registered?", ["Scaffold.errorhandler"]),
    ("How does Flask turn a view's return value into a response object?", ["Flask.make_response"]),
    ("What is the WSGI entry point of the app?", ["Flask.wsgi_app"]),
    ("How does the app dispatch a request to a view function?",
     ["Flask.dispatch_request", "Flask.full_dispatch_request"]),
    ("How do I register a blueprint?", ["Flask.register_blueprint", "App.register_blueprint"]),
    ("How do I get a test client?", ["Flask.test_client"]),
    ("How are JSON responses created?", ["jsonify"]),
    ("How do I redirect to another URL?", ["redirect"]),
    ("How does Flask send a file to the client?", ["send_file", "send_from_directory"]),
    ("How do I abort a request with an HTTP error?", ["abort"]),
    ("How is configuration loaded from a Python file?", ["Config.from_pyfile", "Config.from_object"]),
    ("How do I run the development server?", ["Flask.run"]),
    ("Where are teardown functions called after a request?", ["Flask.do_teardown_request"]),
]

K = 6


def keep(chunk):
    p = chunk["metadata"]["file_path"]
    return not is_test_file(p) and not p.endswith(".md")


def first_rank(chunk_list, expected):
    rank = 0
    for c in chunk_list:
        if not keep(c):
            continue
        rank += 1
        if c["metadata"]["symbol_name"] in expected:
            return rank
        if rank >= K:
            break
    return None


def report(name, ranks):
    hits = [r for r in ranks if r]
    mrr = sum(1 / r for r in hits) / len(ranks)
    print(f"{name:12} hit@{K}: {len(hits)}/{len(ranks)} ({len(hits)/len(ranks):.0%})   MRR: {mrr:.2f}")


if __name__ == "__main__":
    index, chunks = load_index()
    symbols = {c["metadata"]["symbol_name"] for c in chunks}

    for q, exp in TESTS:
        if not any(e in symbols for e in exp):
            print(f"WARNING: none of {exp} found in the index (fix this test)")

    hybrid = HybridSearcher(index, chunks)
    vec_ranks, hyb_ranks, rr_ranks = [], [], []

    print()
    for q, exp in TESTS:
        v = [c for _, c in search(q, index, chunks, k=40)]
        hh = hybrid.search(q, k=40)
        h = [c for _, c in hh]

        pool = [(s, c) for s, c in hh if keep(c)][:30]
        r = [c for _, c in rerank(q, pool, k=len(pool))]

        rv, rh, rr = first_rank(v, exp), first_rank(h, exp), first_rank(r, exp)
        vec_ranks.append(rv)
        hyb_ranks.append(rh)
        rr_ranks.append(rr)
        print(f"vec={str(rv):5} hyb={str(rh):5} rerank={str(rr):5} {q}")

        if rr is None:
            top = [c["metadata"]["symbol_name"] for c in r[:4]]
            print(f"      expected {exp}, got top: {top}")

    print()
    report("vector-only", vec_ranks)
    report("hybrid", hyb_ranks)
    report("hyb+rerank", rr_ranks)