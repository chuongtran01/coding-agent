"""Check that the agent's sandbox cannot reach the answer before evaluation.

Needs Docker and the dev task images; see test_sandbox.py.
"""

import re
import shlex

import pytest

from agent.harness.dataset import Task, load_dev_tasks
from agent.harness.patch import apply_patch
from agent.harness.sandbox import DEFAULT_LIFETIME_S, Sandbox

DEV_TASKS = load_dev_tasks()
_FETCH_GITHUB = (
    'python -c "import urllib.request; '
    "urllib.request.urlopen('https://github.com', timeout=5)\""
)


def _changed_files(diff: str) -> list[str]:
    return re.findall(r"^diff --git a/\S+ b/(\S+)", diff, re.M)


def _new_lines(diff: str, base: str) -> list[str]:
    """The longest lines a diff adds that are not already in the base files.

    Long lines are distinctive; short ones (imports, `result = testdir.runpytest()`)
    turn up all over the image and say nothing about a leak.
    """
    added = {
        line[1:].strip()
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    }
    new = [line for line in added if len(line) > 25 and line not in base]
    return sorted(new, key=len, reverse=True)[:2]


@pytest.mark.parametrize("task", DEV_TASKS, ids=lambda t: t.instance_id)
def test_fixed_files_are_not_in_git(task: Task) -> None:
    """No git object in the image holds a file as it looks after the fix or test patch."""
    files = _changed_files(task.gold_patch) + _changed_files(task.test_patch)
    with Sandbox.start(task) as sb:
        assert apply_patch(sb, task.gold_patch).applied
        assert apply_patch(sb, task.test_patch).applied
        hashes = sb.exec("git hash-object " + " ".join(files)).output.split()
    assert len(hashes) == len(files)
    with Sandbox.start(task) as sb:
        for path, blob in zip(files, hashes):
            assert sb.exec(f"git cat-file -e {blob}").exit_code != 0, path


@pytest.mark.parametrize("task", DEV_TASKS, ids=lambda t: t.instance_id)
def test_no_future_history_or_remotes(task: Task) -> None:
    with Sandbox.start(task) as sb:
        assert sb.exec("git remote").output.strip() == ""
        all_commits = sb.exec("git rev-list --all | wc -l").output.strip()
        head_commits = sb.exec("git rev-list HEAD | wc -l").output.strip()
        unreachable = sb.exec("git fsck --unreachable --no-reflogs", timeout=300)
    assert all_commits == head_commits  # every ref is an ancestor of HEAD
    assert "unreachable" not in unreachable.output


_KNOWN_REQUESTS_LEAK = pytest.mark.xfail(
    reason="images ship newer requests releases that contain the fix", strict=True
)


@pytest.mark.parametrize(
    "task",
    [
        pytest.param(t, marks=_KNOWN_REQUESTS_LEAK) if t.repo == "psf/requests" else t
        for t in DEV_TASKS
    ],
    ids=lambda t: t.instance_id,
)
def test_patch_lines_are_not_on_disk(task: Task) -> None:
    """The gold and test patches' most distinctive lines appear nowhere on disk.

    Known gap: images also hold newer releases of `requests` (pip vendors one), which
    contain some upstream fixes for psf/requests tasks. See docs/DEV_TASKS.md.
    """
    with Sandbox.start(task) as sb:
        files = _changed_files(task.gold_patch) + _changed_files(task.test_patch)
        base = sb.exec("cat " + " ".join(files) + " 2>/dev/null").output
        lines = _new_lines(task.gold_patch, base) + _new_lines(task.test_patch, base)
        assert lines
        for line in lines:
            found = sb.exec(
                "grep -rlF --exclude-dir=proc --exclude-dir=sys --exclude-dir=dev "
                f"-- {shlex.quote(line)} / 2>/dev/null",
                timeout=600,
            )
            assert found.output.strip() == "", line


def test_network_is_off_by_default() -> None:
    with Sandbox.start(DEV_TASKS[0]) as sb:
        assert sb.exec(_FETCH_GITHUB, timeout=30).exit_code != 0


def test_network_can_be_turned_on_for_evaluation() -> None:
    with Sandbox.start(DEV_TASKS[0], network=True) as sb:
        assert sb.exec(_FETCH_GITHUB, timeout=30).exit_code == 0


def test_container_has_bounded_lifetime() -> None:
    with Sandbox.start(DEV_TASKS[0]) as sb:
        sb._container.reload()
        assert sb._container.attrs["Config"]["Cmd"] == [
            "sleep",
            str(DEFAULT_LIFETIME_S),
        ]
    with Sandbox.start(DEV_TASKS[0], lifetime_s=5) as sb:
        sb.exec("sleep 10", timeout=30)
        sb._container.reload()
        assert sb._container.status == "exited"
