import subprocess
from pathlib import Path

import pytest

from agent.context import GrepMatch, RepoError, RepoView

from .shell import LocalShell


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    git(repo, "init", "-q")
    (repo / "pkg" / "models.py").write_text(
        "class Response:\n    def iter_content(self):\n        pass\n"
    )
    (repo / "pkg" / "old.py").write_text("x = 1\n\x0cdef iter_content_old(): pass\n")
    (repo / "odd: name.py").write_text("iter_content = None\n")
    (repo / "README.md").write_text("iter_content docs\n")
    (repo / "blob.py").write_bytes(b"\x00\x01iter_content")
    git(repo, "add", "-A")
    (repo / "untracked.py").write_text("iter_content\n")
    return repo


@pytest.fixture
def view(repo: Path) -> RepoView:
    return RepoView(LocalShell(repo))


def test_python_files_are_tracked_only(view: RepoView) -> None:
    assert view.python_files() == [
        "blob.py",
        "odd: name.py",
        "pkg/models.py",
        "pkg/old.py",
    ]


def test_read(view: RepoView) -> None:
    assert view.read("pkg/models.py").startswith("class Response:\n")
    assert view.read("./pkg/../pkg/models.py").startswith("class Response:")


@pytest.mark.parametrize("path", ["/etc/passwd", "../x.py", "pkg/../../x.py"])
def test_read_refuses_paths_outside_the_repo(view: RepoView, path: str) -> None:
    with pytest.raises(RepoError, match="not a path inside the repo"):
        view.read(path)


def test_read_missing_file_raises(view: RepoView) -> None:
    with pytest.raises(RepoError, match="No such file"):
        view.read("pkg/nope.py")


def test_grep_python_files(view: RepoView) -> None:
    matches = view.grep(["iter_content"])
    # Tracked .py files only, binary files skipped, odd paths and form feeds intact.
    assert matches == [
        GrepMatch(path="odd: name.py", line=1, text="iter_content = None"),
        GrepMatch(path="pkg/models.py", line=2, text="    def iter_content(self):"),
        GrepMatch(path="pkg/old.py", line=2, text="\x0cdef iter_content_old(): pass"),
    ]


def test_grep_options(view: RepoView) -> None:
    assert [m.path for m in view.grep(["iter_content"], word=True)] == [
        "odd: name.py",
        "pkg/models.py",
    ]
    assert [m.line for m in view.grep([r"^(class|def) "], regex=True)] == [1]
    assert view.grep(["RESPONSE"], ignore_case=True)[0].path == "pkg/models.py"
    assert view.grep(["iter_", "content"], only_matching=True)[2:4] == [
        GrepMatch(path="pkg/models.py", line=2, text="iter_"),
        GrepMatch(path="pkg/models.py", line=2, text="content"),
    ]
    assert [m.path for m in view.grep(["docs"], pathspec="*.md")] == ["README.md"]


def test_grep_no_match_or_no_patterns(view: RepoView) -> None:
    assert view.grep(["no such text"]) == []
    assert view.grep([]) == []


def test_grep_error_raises(view: RepoView) -> None:
    with pytest.raises(RepoError, match="failed"):
        view.grep(["("], regex=True)


def test_nothing_is_written(view: RepoView, repo: Path) -> None:
    before = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True
    ).stdout
    view.python_files(), view.read("pkg/models.py"), view.grep(["x"])
    after = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True
    ).stdout
    assert after == before
