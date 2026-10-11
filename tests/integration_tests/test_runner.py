"""Needs Docker and the dev task images; see test_sandbox.py."""

import pytest

from agent.harness.dataset import Task, load_dev_tasks
from agent.harness.patch import apply_patch
from agent.harness.runner import run_tests
from agent.harness.sandbox import Sandbox

DEV_TASKS = load_dev_tasks()


def _passed(status: str | None) -> bool:
    return status in ("PASSED", "XFAIL")


@pytest.mark.parametrize("task", DEV_TASKS, ids=lambda t: t.instance_id)
def test_gold_patch_passes_listed_tests(task: Task) -> None:
    with Sandbox.start(task) as sb:
        assert apply_patch(sb, task.gold_patch).applied
        result = run_tests(sb, task)
    assert result.valid and not result.timed_out
    for test in task.FAIL_TO_PASS + task.PASS_TO_PASS:
        assert _passed(result.statuses.get(test)), test


@pytest.mark.parametrize("task", DEV_TASKS, ids=lambda t: t.instance_id)
def test_empty_patch_fails_fail_to_pass(task: Task) -> None:
    with Sandbox.start(task) as sb:
        result = run_tests(sb, task)
    assert result.valid
    assert not any(_passed(result.statuses.get(t)) for t in task.FAIL_TO_PASS)


def test_timeout_is_reported() -> None:
    task = DEV_TASKS[0]
    with Sandbox.start(task) as sb:
        result = run_tests(sb, task, timeout=1)
    assert result.timed_out
    assert not result.valid
