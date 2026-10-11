import os
import subprocess
import time
from pathlib import Path

import pytest

from agent.harness.sandbox import ExecResult
from agent.tools import MAX_BASH_CHARS, MAX_READ_CHARS, TOOL_NAMES, Toolbox


class LocalShell:
    """Runs commands with the host's bash in a temp dir, standing in for a Sandbox."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def exec(self, cmd: str, timeout: float = 60) -> ExecResult:
        start = time.monotonic()
        try:
            proc = subprocess.run(
                ["bash", "-c", cmd],
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(
                exit_code=124, output="", runtime_s=timeout, timed_out=True
            )
        return ExecResult(
            exit_code=proc.returncode,
            output=proc.stdout.decode("utf-8", errors="replace"),
            runtime_s=time.monotonic() - start,
            timed_out=False,
        )

    def write_file(self, path: str, content: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(
        "def f():\n    return 1\n\n\ndef g():\n    return 1\n"
    )
    return tmp_path


@pytest.fixture
def tb(repo: Path) -> Toolbox:
    return Toolbox(LocalShell(repo), bash_timeout_s=5, workdir=str(repo))


def test_tool_schemas() -> None:
    tools = Toolbox(LocalShell(Path(".")), bash_timeout_s=5).tools()
    assert [t.name for t in tools] == list(TOOL_NAMES)
    edit = next(t for t in tools if t.name == "edit")
    assert set(edit.args) == {"path", "old", "new"}


def test_langchain_tool_goes_through_run(tb: Toolbox) -> None:
    bash = next(t for t in tb.tools() if t.name == "bash")
    assert bash.invoke({"command": "echo hi"}) == "exit code: 0\nhi\n"


def test_read_file_with_line_numbers(tb: Toolbox, repo: Path) -> None:
    out = tb.run("read", {"path": "pkg/mod.py"})
    assert out.startswith(f"{repo}/pkg/mod.py (6 lines)\n")
    assert "     2\t    return 1" in out


def test_read_range(tb: Toolbox) -> None:
    out = tb.run("read", {"path": "pkg/mod.py", "start_line": 5, "end_line": 6})
    assert "     5\tdef g():" in out
    assert "def f" not in out


def test_read_range_errors(tb: Toolbox) -> None:
    assert "out of range" in tb.run("read", {"path": "pkg/mod.py", "start_line": 99})
    assert "before start_line" in tb.run(
        "read", {"path": "pkg/mod.py", "start_line": 3, "end_line": 2}
    )


def test_read_directory(tb: Toolbox) -> None:
    out = tb.run("read", {"path": "."})
    assert "pkg/" in out.splitlines()


def test_read_missing(tb: Toolbox) -> None:
    out = tb.run("read", {"path": "nope.py"})
    assert out.startswith("Error: cannot read") and "No such file" in out


def test_read_caps_long_files(tb: Toolbox, repo: Path) -> None:
    (repo / "big.py").write_text("x = 1  # padding padding padding\n" * 5000)
    out = tb.run("read", {"path": "big.py"})
    assert len(out) < MAX_READ_CHARS + 500
    assert "output capped. Use start_line=" in out


def test_read_binary(tb: Toolbox, repo: Path) -> None:
    (repo / "blob").write_bytes(b"\x00\x01\x02")
    assert "binary" in tb.run("read", {"path": "blob"})


def test_write_creates_dirs(tb: Toolbox, repo: Path) -> None:
    out = tb.run("write", {"path": "new/dir/x.py", "content": "a\nb\n"})
    assert out.endswith("(2 lines).")
    assert (repo / "new/dir/x.py").read_text() == "a\nb\n"


def test_write_keeps_file_mode(tb: Toolbox, repo: Path) -> None:
    script = repo / "run.sh"
    script.write_text("echo old\n")
    script.chmod(0o755)
    tb.run("write", {"path": "run.sh", "content": "echo new\n"})
    assert script.read_text() == "echo new\n"
    assert os.stat(script).st_mode & 0o777 == 0o755


def test_edit_unique(tb: Toolbox, repo: Path) -> None:
    out = tb.run(
        "edit",
        {
            "path": "pkg/mod.py",
            "old": "def f():\n    return 1",
            "new": "def f():\n    return 2",
        },
    )
    assert out.startswith("Edited")
    assert "     2\t    return 2" in out
    assert (repo / "pkg/mod.py").read_text().startswith("def f():\n    return 2\n")


def test_edit_not_unique(tb: Toolbox, repo: Path) -> None:
    before = (repo / "pkg/mod.py").read_text()
    out = tb.run("edit", {"path": "pkg/mod.py", "old": "return 1", "new": "return 2"})
    assert "occurs 2 times" in out
    assert (repo / "pkg/mod.py").read_text() == before


def test_edit_not_found(tb: Toolbox) -> None:
    out = tb.run("edit", {"path": "pkg/mod.py", "old": "\treturn 1", "new": "x"})
    assert "not found" in out and "indentation" in out


def test_edit_rejects_empty_and_noop(tb: Toolbox) -> None:
    assert "empty" in tb.run("edit", {"path": "pkg/mod.py", "old": "", "new": "x"})
    assert "same" in tb.run("edit", {"path": "pkg/mod.py", "old": "f", "new": "f"})


def test_bash_reports_exit_code(tb: Toolbox) -> None:
    assert (
        tb.run("bash", {"command": "echo oops >&2; exit 3"}) == "exit code: 3\noops\n"
    )


def test_bash_runs_in_workdir(tb: Toolbox, repo: Path) -> None:
    assert tb.run("bash", {"command": "pwd"}).splitlines()[1] == str(repo.resolve())


def test_bash_timeout(tb: Toolbox) -> None:
    assert "timed out after 5s" in tb.run("bash", {"command": "sleep 30"})


def test_bash_truncates_from_the_middle(tb: Toolbox) -> None:
    out = tb.run("bash", {"command": "echo START; seq 1 100000; echo END"})
    assert "START" in out and out.rstrip().endswith("END")
    assert "characters cut from the middle" in out
    assert len(out) < MAX_BASH_CHARS + 200


def test_submit(tb: Toolbox) -> None:
    assert tb.run("submit", {}) == "Submitted."


def test_errors_are_returned_not_raised(tb: Toolbox) -> None:
    assert tb.run("rm", {}).startswith("Error: unknown tool 'rm'")
    out = tb.run("edit", {"path": "pkg/mod.py"})
    assert out.startswith("Error: invalid arguments for edit") and "old" in out


def test_tool_exceptions_are_returned(repo: Path) -> None:
    class Broken(LocalShell):
        def exec(self, cmd: str, timeout: float = 60) -> ExecResult:
            raise ConnectionError("docker went away")

    out = Toolbox(Broken(repo), bash_timeout_s=5).run("bash", {"command": "ls"})
    assert out == "Error: bash failed: ConnectionError: docker went away"
