import subprocess
from pathlib import Path

import pytest

from agent.context import RepoView, changed_files, issue_terms, rank_files, recall
from agent.context.rank import CODE_ONLY, PROSE, is_test_file, resolve_path
from agent.harness.dataset import AgentTask

from .shell import LocalShell

FILES = [
    "requests/__init__.py",
    "requests/models.py",
    "requests/utils.py",
    "requests/packages/urllib3/util.py",
    "tests/test_requests.py",
]

ISSUE = """\
`iter_content(decode_unicode=True)` returns bytes, see requests/models.py.

```python
import requests
r = requests.get(url, stream=True)
next(r.iter_content(16, decode_unicode=True))
```

Traceback:
```
  File "/usr/lib/python3/site-packages/requests/utils.py", line 10, in get_encoding
  File "/usr/lib/python3/site-packages/requests/models.py", line 20, in iter_content
```
Maybe `requests.utils.get_encoding_from_headers` or PolyFit is wrong.
"""


def test_issue_terms() -> None:
    terms = issue_terms(ISSUE)
    assert terms.frames == [
        "/usr/lib/python3/site-packages/requests/utils.py",
        "/usr/lib/python3/site-packages/requests/models.py",
    ]
    assert "requests.utils.get_encoding_from_headers" in terms.modules
    ids = terms.identifiers
    assert ids["iter_content"] == 1.0 and ids["decode_unicode"] == 1.0  # inline code
    assert ids["PolyFit"] == 1.0  # CamelCase in prose
    assert ids["get_encoding"] == CODE_ONLY  # only in a code block
    assert ids["get"] == CODE_ONLY  # a call
    # Python's own names, path parts and short words are not terms.
    for name in ("True", "import", "self", "lib", "site", "r", "py"):
        assert name not in ids


def test_capitalised_dotted_heads_are_classes_and_dunders_are_dropped() -> None:
    ids = issue_terms(
        "TypeError: RST.__init__() got an unexpected argument"
    ).identifiers
    assert "RST" in ids
    assert "__init__" not in ids
    assert "tbl" not in issue_terms("calling tbl.write fails").identifiers


def test_options_hyphenated_names_and_exception_messages() -> None:
    issue = (
        "Show scopes with `pytest --fixtures` and handle an empty If-Modified-Since."
        "\n```\nTypeError: RST.__init__() got an unexpected keyword 'header_rows'\n```"
    )
    ids = issue_terms(issue).identifiers
    assert ids["--fixtures"] == 1.0
    assert ids["If-Modified-Since"] == 1.0 and ids["if_modified_since"] == CODE_ONLY
    assert ids["RST"] == 1.0 and ids["header_rows"] == 1.0  # not CODE_ONLY
    assert ids["scopes"] == PROSE


def test_frames_are_not_paths_named_in_the_issue() -> None:
    terms = issue_terms(ISSUE)
    assert terms.paths == ["requests/models.py"]  # from the prose only


def test_issue_terms_frame_styles() -> None:
    text = (
        "File ~/code/seaborn/seaborn/_core/plot.py:265, in Plot._repr_png_(self)\n"
        "SKIPPED [1] src/_pytest/skipping.py:238: unconditional skip\n"
    )
    assert issue_terms(text).frames == [
        "~/code/seaborn/seaborn/_core/plot.py",
        "src/_pytest/skipping.py",
    ]


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("requests/models.py", ["requests/models.py"]),
        ("./requests/models.py", ["requests/models.py"]),
        ("/opt/env/site-packages/requests/models.py", ["requests/models.py"]),
        ("models.py", ["requests/models.py"]),
        ("util.py", ["requests/packages/urllib3/util.py"]),
        ("__init__.py", ["requests/__init__.py"]),
        ("other/models.py", []),
        ("test_it.py", []),
    ],
)
def test_resolve_path(token: str, expected: list[str]) -> None:
    assert resolve_path(token, FILES) == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tests/test_x.py", True),
        ("src/testing/helpers.py", True),
        ("pkg/test_x.py", True),
        ("pkg/x_test.py", True),
        ("conftest.py", True),
        ("pkg/testing_utils.py", False),
        ("src/_pytest/skipping.py", False),
    ],
)
def test_is_test_file(path: str, expected: bool) -> None:
    assert is_test_file(path) is expected


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def view(tmp_path: Path) -> RepoView:
    files = {
        "pkg/models.py": "class Response:\n    def iter_content(self):\n        pass\n",
        "pkg/utils.py": "def helper():\n    return iter_content\n",
        "pkg/frames.py": "def outer():\n    inner()\n",
        "pkg/inner.py": "def inner():\n    raise ValueError\n",
        "pkg/config.py": "DEFAULT_FORMAT = '%(name)s'\n",
        "tests/test_models.py": "def test_iter_content():\n    iter_content()\n",
    }
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


def test_a_definition_beats_a_mention(view: RepoView) -> None:
    ranked = rank_files(task("`iter_content` is broken"), view)
    assert [f.path for f in ranked] == [
        "pkg/models.py",
        "pkg/utils.py",
        "tests/test_models.py",  # mentions it twice, but tests score half
    ]
    assert ranked[0].reasons[0] == "defines iter_content"


def test_a_named_path_wins(view: RepoView) -> None:
    ranked = rank_files(task("`iter_content` fails, see pkg/utils.py"), view)
    assert ranked[0].path == "pkg/utils.py"
    assert ranked[0].reasons[0] == "path in issue"


def test_module_level_constants_are_definitions(view: RepoView) -> None:
    ranked = rank_files(task("Change `DEFAULT_FORMAT`"), view)
    assert ranked[0].path == "pkg/config.py"
    assert ranked[0].reasons[0] == "defines DEFAULT_FORMAT"


def test_the_innermost_frame_scores_most(view: RepoView) -> None:
    issue = (
        "```\n"
        '  File "/src/pkg/frames.py", line 2, in outer\n'
        '  File "/src/pkg/inner.py", line 2, in inner\n'
        "ValueError\n```"
    )
    ranked = rank_files(task(issue), view)
    assert [f.path for f in ranked[:2]] == ["pkg/inner.py", "pkg/frames.py"]
    assert "last repo frame in traceback" in ranked[0].reasons


def test_ranking_is_deterministic(view: RepoView) -> None:
    issue = "`iter_content` and `Response` and `helper` break pkg/frames.py"
    assert rank_files(task(issue), view) == rank_files(task(issue), view)


def test_no_terms_ranks_nothing(view: RepoView) -> None:
    assert rank_files(task("it is broken"), view) == []


def test_changed_files_and_recall() -> None:
    diff = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
        "diff --git a/old.py b/new.py\nrename from old.py\n"
        "diff --git a/a.py b/a.py\n"
    )
    assert changed_files(diff) == ["a.py", "new.py"]
    assert recall(["a.py", "x.py"], ["a.py", "new.py"]) == 0.5
    assert recall(["a.py"], []) is None


def test_a_prose_issue_ranks_by_file_name(tmp_path: Path) -> None:
    files = {
        "src/app/blueprints.py": "class Blueprint:\n    pass\n",
        "src/app/indexentries.py": "def create():\n    pass\n",
        "src/app/utils/__init__.py": "x = 1\n",
        "src/app/a/utils.py": "",
        "src/app/b/utils.py": "",
        "src/app/c/utils.py": "",
        "src/app/d/utils.py": "",
    }
    for path, text in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    view = RepoView(LocalShell(tmp_path))

    ranked = rank_files(task("Raise an error when a blueprint name has a dot"), view)
    assert ranked[0].path == "src/app/blueprints.py"
    assert ranked[0].reasons[0] == "name matches 'blueprint'"
    ranked = rank_files(task("Two sections for symbols in index entries"), view)
    assert ranked[0].path == "src/app/indexentries.py"
    # A name that fits five files is too common to say anything.
    assert rank_files(task("broken utils everywhere"), view) == []
