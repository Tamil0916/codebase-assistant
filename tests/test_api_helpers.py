import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from fastapi import HTTPException

from main import repo_id_from_url


def test_valid_github_url_becomes_repo_id():
    assert repo_id_from_url("https://github.com/pallets/flask") == "pallets__flask"


def test_trailing_slash_and_dot_git_are_accepted():
    assert repo_id_from_url("https://github.com/psf/requests.git") == "psf__requests"
    assert repo_id_from_url("https://github.com/psf/requests/") == "psf__requests"


@pytest.mark.parametrize("bad", [
    "http://github.com/pallets/flask",
    "https://gitlab.com/pallets/flask",
    "https://evil.com/https://github.com/pallets/flask",
    "file:///C:/secrets",
    "not a url",
])
def test_non_github_urls_are_rejected(bad):
    with pytest.raises(HTTPException):
        repo_id_from_url(bad)