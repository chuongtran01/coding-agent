"""Decide whether a test run resolves its task, using SWE-bench's own grading."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from swebench.harness.constants import (  # type: ignore[import-untyped]
    FAIL_TO_PASS,
    PASS_TO_PASS,
    EvalType,
    ResolvedStatus,
)
from swebench.harness.grading import (  # type: ignore[import-untyped]
    get_eval_tests_report,
    get_resolution_status,
)

from agent.harness.dataset import Task
from agent.harness.runner import TestResult

Resolution = Literal["RESOLVED_FULL", "RESOLVED_PARTIAL", "RESOLVED_NO"]


class TestOutcomes(BaseModel):
    """Listed tests split by whether they ended up where the task needs them."""

    success: list[str]
    failure: list[str]


class Score(BaseModel):
    """Verdict for one run, with the per-test breakdown behind it.

    Only `RESOLVED_FULL` counts as resolved. `RESOLVED_PARTIAL` (some but not all
    FAIL_TO_PASS tests pass, no regressions) is reported for insight, as SWE-bench does.
    """

    resolved: bool
    status: Resolution
    fail_to_pass: TestOutcomes
    pass_to_pass: TestOutcomes


def score(task: Task, result: TestResult) -> Score:
    """Grade `result` against the task's FAIL_TO_PASS and PASS_TO_PASS lists.

    Mirrors SWE-bench's `get_eval_report`: an invalid run resolves nothing, and a
    listed test that is missing from the log counts as a failure.
    """
    if not result.valid:
        return Score(
            resolved=False,
            status=ResolvedStatus.NO.value,
            fail_to_pass=TestOutcomes(success=[], failure=list(task.FAIL_TO_PASS)),
            pass_to_pass=TestOutcomes(success=[], failure=list(task.PASS_TO_PASS)),
        )

    expected = {
        "instance_id": task.instance_id,
        FAIL_TO_PASS: list(task.FAIL_TO_PASS),
        PASS_TO_PASS: list(task.PASS_TO_PASS),
    }
    report = get_eval_tests_report(
        dict(result.statuses), expected, eval_type=EvalType(task.eval_type)
    )
    status = get_resolution_status(report)
    return Score(
        resolved=status == ResolvedStatus.FULL.value,
        status=status,
        fail_to_pass=TestOutcomes(**report[FAIL_TO_PASS]),
        pass_to_pass=TestOutcomes(**report[PASS_TO_PASS]),
    )


def is_resolved(task: Task, result: TestResult) -> bool:
    """Return True if every FAIL_TO_PASS test passes and every PASS_TO_PASS test still does."""
    return score(task, result).resolved
