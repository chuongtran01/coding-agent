from pathlib import Path

from agent.context import get_strategy
from agent.context.skeleton import skeleton

from .test_full_file import BIG, task, view

SOURCE = '''\
import os
from typing import (
    Any,
)


@decorated
class Response(Base):
    """An HTTP response.

    More words.
    """

    encoding = None

    def iter_content(
        self, decode_unicode=False
    ):
        """
        Iterate over the body.
        """
        yield 1

    async def fetch(self): return 1


def helper():
    return 2
'''


def shown(text: str) -> list[int]:
    return [int(row.split("\t")[0]) for row in text.splitlines()[1:]]


def test_skeleton_shows_imports_signatures_and_first_docstring_lines() -> None:
    result = skeleton("m.py", SOURCE, {})
    assert result is not None and not result.trimmed and not result.fallback
    assert result.text.startswith("m.py (28 lines; skeleton: imports, signatures")
    assert shown(result.text) == [1, 2, 3, 4, 7, 8, 9, 16, 17, 18, 20, 24, 27]
    assert "..." not in result.text  # line numbers show the gaps
    assert "yield" not in result.text and "More words" not in result.text


def test_a_skeleton_too_big_keeps_the_best_matches() -> None:
    full = skeleton("m.py", SOURCE, {"helper": 1.0})
    assert full is not None
    cut = skeleton("m.py", SOURCE, {"helper": 1.0}, len(full.text) - 1)
    assert cut is not None and cut.trimmed and len(cut.text) < len(full.text)
    assert "def helper" in cut.text


def test_a_file_that_does_not_parse_falls_back_to_def_and_class_lines() -> None:
    text = "import os\n\nclass A:\n    def f(self):\n        print 'hi'\n"
    result = skeleton("m.py", text, {})
    assert result is not None and result.fallback and not result.trimmed
    assert shown(result.text) == [1, 3, 4]
    assert "it does not parse" in result.text.splitlines()[0]


def test_a_file_without_definitions_has_no_skeleton() -> None:
    assert skeleton("m.py", "x = 1\nprint(x)\n", {}) is None


def test_repo_map_covers_the_ranked_files_with_skeletons(tmp_path: Path) -> None:
    repo = view(
        tmp_path,
        {
            "pkg/models.py": BIG,
            "pkg/other.py": "def helper():\n    return iter_content\n",
            "pkg/unrelated.py": "def f():\n    pass\n",  # so the mention is rare
        },
    )
    context = get_strategy("repo_map").build(
        task("`iter_content` is broken"), repo, 24_000
    )
    assert [i.path for i in context.items] == ["pkg/models.py", "pkg/other.py"]
    assert {i.kind for i in context.items} == {"skeleton"}
    assert "def iter_content(self, decode_unicode=False):" in context.text
    assert "yield stream_decode" not in context.text  # no bodies
    assert sum(i.chars for i in context.items) + 2 == len(context.text)


def test_repo_map_stays_within_the_budget(tmp_path: Path) -> None:
    padded = BIG + "".join(
        f"\n\ndef filler_{i}():\n    return {i}\n" for i in range(80)
    )
    repo = view(tmp_path, {"pkg/models.py": padded})
    full = skeleton("pkg/models.py", padded, {})
    assert full is not None
    budget = len(full.text) - 1
    context = get_strategy("repo_map").build(
        task("`iter_content` is broken"), repo, budget
    )
    (item,) = context.items
    assert item.trimmed and len(context.text) <= budget // 2
    assert "def iter_content" in context.text
