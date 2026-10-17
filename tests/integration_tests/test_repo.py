"""Needs Docker and the first dev task's image (psf__requests-3362)."""

from collections.abc import Iterator

import pytest

from agent.context import RepoView
from agent.harness.dataset import load_dev_tasks
from agent.harness.sandbox import Sandbox

TASK = load_dev_tasks()[0]


@pytest.fixture(scope="module")
def view() -> Iterator[RepoView]:
    with Sandbox.start(TASK) as sb:
        yield RepoView(sb)


def test_python_files(view: RepoView) -> None:
    files = view.python_files()
    assert "requests/models.py" in files
    assert all(f.endswith(".py") and not f.startswith("/") for f in files)


def test_read_matches_grep(view: RepoView) -> None:
    (match,) = view.grep(["def iter_content("], pathspec="requests/models.py")
    lines = view.read("requests/models.py").split("\n")
    assert lines[match.line - 1] == match.text


def test_grep_finds_definitions_across_the_repo(view: RepoView) -> None:
    paths = {m.path for m in view.grep(["def stream_decode_response_unicode"])}
    assert paths == {"requests/utils.py"}
