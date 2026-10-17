import subprocess
from pathlib import Path

from agent.context import RepoView, get_strategy
from agent.context.excerpt import GAP, excerpt, numbered, whole
from agent.harness.dataset import AgentTask

from .shell import LocalShell

BIG = '''\
import os
from typing import Any


@decorated
class Response:
    """An HTTP response."""

    encoding = None

    def iter_content(self, decode_unicode=False):
        for chunk in self.raw:
            yield stream_decode(chunk)

    def json(self) -> Any:
        return parse(self.text)


def stream_decode(chunk):
    return chunk.decode()


def unrelated():
    return 1
'''


HEADER = (
    "m.py (6 lines; showing its imports and the definitions that match the issue)\n"
)


def test_whole_numbers_lines_like_read() -> None:
    assert (
        whole("a.py", "x = 1\ny = 2\n")
        == "a.py (2 lines)\n     1\tx = 1\n     2\ty = 2"
    )


def test_numbered_marks_gaps() -> None:
    lines = ["a", "b", "c", "d"]
    assert numbered(lines, {1, 3, 4}) == f"     1\ta\n{GAP}\n     3\tc\n     4\td"


def test_excerpt_shows_imports_and_matching_definitions_whole() -> None:
    text = excerpt("m.py", BIG, {"iter_content": 1.0, "stream_decode": 1.0}, 10_000)
    assert text is not None
    assert text.startswith("m.py (24 lines; showing its imports")
    shown = {int(row.split("\t")[0]) for row in text.splitlines()[1:] if "\t" in row}
    assert {1, 2} <= shown  # imports
    assert {5, 6} <= shown  # the class's decorator and line, not its docstring
    assert set(range(11, 14)) <= shown  # iter_content, whole
    assert {19, 20} <= shown  # stream_decode
    assert not {7, 9, 15, 16, 23, 24} & shown  # unmatched definitions


def test_excerpt_leaves_out_what_does_not_fit_and_never_cuts_a_function() -> None:
    terms = {"iter_content": 1.0, "stream_decode": 1.0}
    full = excerpt("m.py", BIG, terms, 10_000)
    assert full is not None
    # Room for the header, imports and stream_decode, but not iter_content too.
    text = excerpt("m.py", BIG, terms, len(full) - 40)
    assert text is not None and len(text) <= len(full) - 40
    assert "def stream_decode" in text or "def iter_content" in text
    for name, body in (("iter_content", "yield"), ("stream_decode", "decode()")):
        assert (f"def {name}" in text) == (body in text)


def test_excerpt_needs_a_match_and_parsable_code() -> None:
    assert excerpt("m.py", BIG, {"nothing_here": 1.0}, 10_000) is None
    assert excerpt("m.py", "print 'python 2'\n", {"print": 1.0}, 10_000) is None
    assert excerpt("m.py", BIG, {"iter_content": 1.0}, 10) is None


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def view(tmp_path: Path, files: dict[str, str]) -> RepoView:
    for path, text in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    return RepoView(LocalShell(tmp_path))


def task(issue: str) -> AgentTask:
    return AgentTask(
        instance_id="x__x-1", repo="x/x", base_commit="a" * 40, problem_statement=issue
    )


def test_full_file_shows_ranked_files_whole_within_budget(tmp_path: Path) -> None:
    repo = view(
        tmp_path,
        {
            "pkg/models.py": BIG,
            "pkg/other.py": "def helper():\n    return iter_content\n",
            "pkg/empty.py": "",
        },
    )
    context = get_strategy("full_file").build(
        task("`iter_content` is broken"), repo, 24_000
    )
    assert [i.path for i in context.items] == ["pkg/models.py", "pkg/other.py"]
    assert not any(i.trimmed for i in context.items)
    assert context.text.startswith("pkg/models.py (24 lines)\n     1\timport os\n")
    assert "\n\npkg/other.py (2 lines)\n" in context.text
    assert sum(i.chars for i in context.items) + 2 == len(context.text)
    assert context.items[0].reasons[0] == "defines iter_content"


def test_full_file_trims_a_file_too_big_for_the_budget(tmp_path: Path) -> None:
    padded = BIG + "".join(
        f"\n\ndef filler_{i}():\n    return {i}\n" for i in range(40)
    )
    repo = view(tmp_path, {"pkg/models.py": padded})
    budget = len(whole("pkg/models.py", padded)) - 1
    context = get_strategy("full_file").build(
        task("`iter_content` is broken"), repo, budget
    )
    (item,) = context.items
    # A trimmed file takes at most half the room, leaving the rest for other files.
    assert item.trimmed and item.chars == len(context.text) <= budget // 2
    assert "def iter_content" in context.text and "def unrelated" not in context.text


def test_full_file_with_nothing_ranked_is_empty(tmp_path: Path) -> None:
    repo = view(tmp_path, {"pkg/models.py": BIG})
    context = get_strategy("full_file").build(task("It crashes."), repo, 24_000)
    assert context.text == "" and context.items == []


def test_excerpt_ignores_definitions_that_share_only_prose_words() -> None:
    assert excerpt("m.py", BIG, {"chunk": 0.3, "raw": 0.3}, 10_000) is None


def test_a_named_definition_outranks_one_using_many_terms() -> None:
    text = (
        "def busy():\n    return a + b + c + d + e + f\n\n\n"
        "def named():\n    return 0\n"
    )
    terms = {name: 1.0 for name in "abcdef"} | {"named": 1.0}
    room = len(HEADER) + len(numbered(text.splitlines(), {5, 6}))  # one function
    one = excerpt("m.py", text, terms, room)
    assert one is not None and "def named" in one and "def busy" not in one


def test_a_named_class_brings_its_constructor() -> None:
    text = (
        "class Perm:\n    def order(self):\n        return x\n\n"
        "    def __new__(cls, *args):\n        return args\n"
    )
    room = len(HEADER) + len(numbered(text.splitlines(), {1, 5, 6}))
    one = excerpt("m.py", text, {"Perm": 1.0, "x": 1.0}, room)
    assert one is not None and "def __new__" in one and "def order" not in one
