"""Evaluate one candidate patch for one task, end to end."""

from __future__ import annotations

import time

from pydantic import BaseModel

from agent.harness.dataset import Task
from agent.harness.patch import ApplyResult, apply_patch
from agent.harness.runner import DEFAULT_TEST_TIMEOUT_S, TestResult, run_tests
from agent.harness.sandbox import Sandbox
from agent.harness.scoring import Score, score


class Evaluation(BaseModel):
    """Everything known about one evaluation run.

    `tests` and `score` are None when the patch did not apply; the run stops there.
    """

    instance_id: str
    resolved: bool
    apply: ApplyResult
    tests: TestResult | None
    score: Score | None
    runtime_s: float  # wall clock for the whole run, container start to teardown


def evaluate(
    task: Task, diff: str, test_timeout: float = DEFAULT_TEST_TIMEOUT_S
) -> Evaluation:
    """Apply `diff` to a fresh sandbox for `task`, run the hidden tests, and score."""
    start = time.monotonic()
    tests: TestResult | None = None
    verdict: Score | None = None
    # Network on: the eval script reinstalls the package, which fetches build tools.
    with Sandbox.start(task, network=True) as sb:
        applied = apply_patch(sb, diff)
        if applied.applied:
            tests = run_tests(sb, task, timeout=test_timeout)
            verdict = score(task, tests)
    return Evaluation(
        instance_id=task.instance_id,
        resolved=verdict is not None and verdict.resolved,
        apply=applied,
        tests=tests,
        score=verdict,
        runtime_s=time.monotonic() - start,
    )
