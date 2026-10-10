"""Needs Docker and the dev task images; see test_sandbox.py."""

import re

import pytest

from agent.harness.dataset import Task, load_dev_tasks
from agent.harness.patch import apply_patch
from agent.harness.sandbox import Sandbox

DEV_TASKS = load_dev_tasks()


@pytest.mark.parametrize("task", DEV_TASKS, ids=lambda t: t.instance_id)
def test_gold_and_test_patch_apply(task: Task) -> None:
    with Sandbox.start(task) as sb:
        assert apply_patch(sb, task.gold_patch).applied
        assert apply_patch(sb, task.test_patch).applied
        changed = sb.exec("git status --porcelain").output
    assert changed.strip()
    assert "agent_patches" not in changed


def test_empty_patch_changes_nothing() -> None:
    with Sandbox.start(DEV_TASKS[0]) as sb:
        assert apply_patch(sb, "").applied
        assert sb.exec("git status --porcelain").output == ""


def test_broken_patch_reports_error_and_leaves_tree_clean() -> None:
    task = DEV_TASKS[0]
    # Change a removed line so the hunk no longer matches the file.
    broken = re.sub(
        r"^-(?!--)(.*)$",
        r"-\1  # not in the file",
        task.gold_patch,
        count=1,
        flags=re.M,
    )
    with Sandbox.start(task) as sb:
        result = apply_patch(sb, broken)
        status = sb.exec("git status --porcelain").output
    assert not result.applied
    assert result.error and "patch failed" in result.error
    assert status == ""


def test_garbage_patch_reports_error() -> None:
    with Sandbox.start(DEV_TASKS[0]) as sb:
        result = apply_patch(sb, "this is not a diff")
    assert not result.applied
    assert result.error


def test_same_patch_twice_fails_second_time() -> None:
    task = DEV_TASKS[0]
    with Sandbox.start(task) as sb:
        assert apply_patch(sb, task.gold_patch).applied
        second = apply_patch(sb, task.gold_patch)
    assert not second.applied
    assert second.error
