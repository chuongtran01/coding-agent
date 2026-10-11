"""Needs Docker and the dev task images; see test_sandbox.py."""

from agent.harness.cli import break_patch
from agent.harness.dataset import load_dev_tasks
from agent.harness.evaluate import evaluate

TASK = load_dev_tasks()[0]


def test_gold_patch_resolves() -> None:
    result = evaluate(TASK, TASK.gold_patch)
    assert result.resolved
    assert result.tests is not None and result.tests.valid
    assert result.runtime_s > result.tests.runtime_s


def test_broken_patch_stops_before_tests() -> None:
    result = evaluate(TASK, break_patch(TASK.gold_patch))
    assert not result.resolved
    assert not result.apply.applied and result.apply.error
    assert result.tests is None and result.score is None
