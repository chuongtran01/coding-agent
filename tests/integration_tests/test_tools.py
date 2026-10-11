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


def run(tb: Toolbox, name: str, **args: object) -> str:
    result = tb.run(name, args)
    assert not result.is_error, result.output
    return result.output


def test_read_repo_root(tb: Toolbox) -> None:
    out = run(tb, "read", path=".")
    assert out.startswith("/testbed/ ")
    assert ".git/" in out.splitlines()


def test_write_edit_read_and_diff(tb: Toolbox) -> None:
    run(tb, "write", path="scratch/x.py", content="a = 1\nb = 2\n")
    assert run(tb, "edit", path="scratch/x.py", old="b = 2", new="b = 3").startswith(
        "Edited"
    )
    assert "     2\tb = 3" in run(tb, "read", path="/testbed/scratch/x.py")
    diff = run(tb, "bash", command="git add -A && git diff --cached HEAD --stat")
    assert diff.startswith("exit code: 0") and "scratch/x.py" in diff
    run(tb, "bash", command="git reset -q && rm -rf scratch")


def test_edit_tracked_file_keeps_mode(tb: Toolbox) -> None:
    path = run(tb, "bash", command="git ls-files '*.py' | head -1").splitlines()[1]
    first = run(tb, "read", path=path, end_line=1).splitlines()[1].split("\t", 1)[1]
    assert run(tb, "edit", path=path, old=first, new=first + "  # x").startswith(
        "Edited"
    )
    diff = run(tb, "bash", command=f"git diff -- {path}")
    assert "+" + first + "  # x" in diff and "old mode" not in diff
    run(tb, "bash", command=f"git checkout -q -- {path}")


def test_bash_uses_task_python_and_no_network(tb: Toolbox) -> None:
    prefix = run(tb, "bash", command="python -c 'import sys; print(sys.prefix)'")
    assert "envs/testbed" in prefix
    fetch = tb.run(
        "bash",
        {
            "command": 'python -c "import urllib.request as u; '
            "u.urlopen('https://pypi.org', timeout=5)\""
        },
    )
    assert fetch.exit_code not in (None, 0)
