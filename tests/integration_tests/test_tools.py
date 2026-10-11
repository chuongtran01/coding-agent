"""Needs Docker and the first dev task's image."""

from collections.abc import Iterator

import pytest

from agent.harness.dataset import load_dev_tasks
from agent.harness.sandbox import Sandbox
from agent.tools import Toolbox

TASK = load_dev_tasks()[0]


@pytest.fixture(scope="module")
def tb() -> Iterator[Toolbox]:
    with Sandbox.start(TASK) as sb:
        yield Toolbox(sb, bash_timeout_s=30)


def test_read_repo_root(tb: Toolbox) -> None:
    out = tb.run("read", {"path": "."})
    assert out.startswith("/testbed/ ")
    assert ".git/" in out.splitlines()


def test_write_edit_read_and_diff(tb: Toolbox) -> None:
    assert tb.run("write", {"path": "scratch/x.py", "content": "a = 1\nb = 2\n"})
    assert tb.run(
        "edit", {"path": "scratch/x.py", "old": "b = 2", "new": "b = 3"}
    ).startswith("Edited")
    assert "     2\tb = 3" in tb.run("read", {"path": "/testbed/scratch/x.py"})
    diff = tb.run("bash", {"command": "git add -A && git diff --cached HEAD --stat"})
    assert diff.startswith("exit code: 0") and "scratch/x.py" in diff
    tb.run("bash", {"command": "git reset -q && rm -rf scratch"})


def test_edit_tracked_file_keeps_mode(tb: Toolbox) -> None:
    path = tb.run("bash", {"command": "git ls-files '*.py' | head -1"}).splitlines()[1]
    first = (
        tb.run("read", {"path": path, "end_line": 1}).splitlines()[1].split("\t", 1)[1]
    )
    assert tb.run(
        "edit", {"path": path, "old": first, "new": first + "  # x"}
    ).startswith("Edited")
    diff = tb.run("bash", {"command": f"git diff -- {path}"})
    assert "+" + first + "  # x" in diff and "old mode" not in diff
    tb.run("bash", {"command": f"git checkout -q -- {path}"})


def test_bash_uses_task_python_and_no_network(tb: Toolbox) -> None:
    assert "envs/testbed" in tb.run(
        "bash", {"command": "python -c 'import sys; print(sys.prefix)'"}
    )
    out = tb.run(
        "bash",
        {
            "command": "python -c \"import urllib.request as u; u.urlopen('https://pypi.org', timeout=5)\""
        },
    )
    assert not out.startswith("exit code: 0")
