"""Needs Docker and the dev task images; see test_sandbox.py."""

import pytest

from agent.harness.dataset import Task, load_dev_tasks
from agent.harness.patch import apply_patch
from agent.harness.runner import run_tests
from agent.harness.sandbox import Sandbox
from agent.harness.scoring import score

DEV_TASKS = load_dev_tasks()


@pytest.mark.parametrize("task", DEV_TASKS, ids=lambda t: t.instance_id)
def test_gold_resolves_and_empty_does_not(task: Task) -> None:
    with Sandbox.start(task) as sb:
        assert apply_patch(sb, task.gold_patch).applied
        gold = score(task, run_tests(sb, task))
    with Sandbox.start(task) as sb:
        empty = score(task, run_tests(sb, task))
    assert gold.resolved, gold
    assert not empty.resolved
    assert empty.fail_to_pass.failure == list(task.FAIL_TO_PASS)
    assert empty.pass_to_pass.failure == []
