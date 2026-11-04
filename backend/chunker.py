import ast
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

LANG_MAP = {
    ".js": Language.JS,
    ".ts": Language.TS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".md": Language.MARKDOWN,
}


def make_chunk(text, path, language, symbol, start, end):
    return {
        "text": text,
        "metadata": {
            "file_path": path,
            "language": language,
            "symbol_name": symbol,
            "start_line": start,
            "end_line": end,
        },
    }


def node_start(node):
    # include decorators in the chunk
    lines = [node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]
    return min(lines)


def add_module_chunks(chunks, lines, path):
    """Index code that sits outside any function/class (constants, messages, setup)."""
    covered = set()
    for c in chunks:
        m = c["metadata"]
        covered.update(range(m["start_line"], m["end_line"] + 1))

    run = []

    def flush():
        while run and not lines[run[0] - 1].strip():
            run.pop(0)
        while run and not lines[run[-1] - 1].strip():
            run.pop()
        if run:
            start, end = run[0], run[-1]
            text = "\n".join(lines[start - 1:end])
            body = [l for l in text.splitlines()
                    if l.strip() and not l.lstrip().startswith(("import ", "from ", "#"))]
            if body:  # skip chunks that are only imports/comments
                chunks.append(make_chunk(text, path, "python", "(module-level)", start, end))
        run.clear()

    for i in range(1, len(lines) + 1):
        if i in covered:
            flush()
        else:
            run.append(i)
    flush()


def chunk_python(path, content):
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return chunk_generic(path, content, ".py")

    lines = content.splitlines()
    chunks = []

    def add(node, symbol, end=None):
        start = node_start(node)
        end = end or node.end_lineno
        text = "\n".join(lines[start - 1:end])
        chunks.append(make_chunk(text, path, "python", symbol, start, end))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node, node.name)

        elif isinstance(node, ast.ClassDef):
            methods = [n for n in node.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            # class header: from "class X:" down to just before the first method
            header_end = (node_start(methods[0]) - 1) if methods else node.end_lineno
            add(node, node.name, end=header_end)
            for m in methods:
                add(m, f"{node.name}.{m.name}")

    # also index module-level code (constants, messages, setup code)
    add_module_chunks(chunks, lines, path)

    # fall back if a file has no functions/classes (e.g. config or scripts)
    if not chunks and content.strip():
        return chunk_generic(path, content, ".py")
    return chunks


def chunk_generic(path, content, ext):
    kwargs = dict(chunk_size=1500, chunk_overlap=150, add_start_index=True)
    lang = LANG_MAP.get(ext)
    if lang:
        splitter = RecursiveCharacterTextSplitter.from_language(language=lang, **kwargs)
    elif ext == ".py":
        splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, **kwargs)
    else:
        splitter = RecursiveCharacterTextSplitter(**kwargs)

    chunks = []
    docs = splitter.create_documents([content])
    for d in docs:
        start = content.count("\n", 0, d.metadata["start_index"]) + 1
        end = start + d.page_content.count("\n")
        chunks.append(make_chunk(d.page_content, path, ext.lstrip("."),
                                 "(section)", start, end))
    return chunks


def chunk_file(path, content, root=""):
    rel = os.path.relpath(path, root) if root else path
    ext = os.path.splitext(path)[1]
    if ext == ".py":
        return chunk_python(rel, content)
    return chunk_generic(rel, content, ext)


def chunk_repo(files, root=""):
    all_chunks = []
    for f in files:
        all_chunks.extend(chunk_file(f["path"], f["content"], root))
    return all_chunks


if __name__ == "__main__":
    from ingest import clone_repo, load_code_files

    root = clone_repo("https://github.com/pallets/flask")
    files = load_code_files(root)
    chunks = chunk_repo(files, root)
    print(f"{len(files)} files -> {len(chunks)} chunks")

    # show module-level chunks as a sanity check
    shown = 0
    for c in chunks:
        m = c["metadata"]
        if m["symbol_name"] == "(module-level)":
            print("-" * 60)
            print(f'{m["file_path"]}  [{m["symbol_name"]}]  lines {m["start_line"]}-{m["end_line"]}')
            print(c["text"][:300])
            shown += 1
            if shown == 3:
                break