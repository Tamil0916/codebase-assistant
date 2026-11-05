import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from chunker import chunk_python

SAMPLE = '''import os

ERROR_MSG = "Working outside of context"


def top_level(a, b):
    """Add two numbers."""
    return a + b


class Greeter:
    """Says hello."""

    def hello(self, name):
        return f"hello {name}"

    @staticmethod
    def bye():
        return "bye"
'''


def names(chunks):
    return [c["metadata"]["symbol_name"] for c in chunks]


def test_functions_and_methods_become_separate_chunks():
    chunks = chunk_python("sample.py", SAMPLE)
    assert "top_level" in names(chunks)
    assert "Greeter.hello" in names(chunks)
    assert "Greeter.bye" in names(chunks)


def test_module_level_code_is_indexed():
    chunks = chunk_python("sample.py", SAMPLE)
    module_chunks = [c for c in chunks if c["metadata"]["symbol_name"] == "(module-level)"]
    assert any("ERROR_MSG" in c["text"] for c in module_chunks)


def test_line_numbers_match_the_source():
    chunks = chunk_python("sample.py", SAMPLE)
    lines = SAMPLE.splitlines()
    for c in chunks:
        m = c["metadata"]
        assert c["text"] == "\n".join(lines[m["start_line"] - 1:m["end_line"]])


def test_decorator_is_included_in_method_chunk():
    chunks = chunk_python("sample.py", SAMPLE)
    bye = next(c for c in chunks if c["metadata"]["symbol_name"] == "Greeter.bye")
    assert "@staticmethod" in bye["text"]


def test_unparseable_file_falls_back_without_crashing():
    chunks = chunk_python("broken.py", "def oops(:\n    pass\n")
    assert len(chunks) >= 1