"""Run a task's hidden tests in a sandbox and parse a status per test."""

from __future__ import annotations

import tempfile
from typing import Any, Literal

from pydantic import BaseModel
from swebench.harness.grading import get_logs_eval  # type: ignore[import-untyped]
from swebench.harness.utils import make_test_spec  # type: ignore[import-untyped]

from agent.harness.dataset import Task
from agent.harness.sandbox import Sandbox

# SWE-bench's own default per-instance timeout.
DEFAULT_TEST_TIMEOUT_S = 1800
# Kept from the end of the log, where the per-test summary is.
MAX_OUTPUT_CHARS = 50_000

# Outside /testbed so it never shows up in the repo's diff.
_EVAL_SCRIPT_PATH = "/tmp/agent_eval/eval.sh"

TestStatus = Literal["PASSED", "FAILED", "SKIPPED", "ERROR", "XFAIL"]


class TestResult(BaseModel):
    """Outcome of one evaluation run.

    `statuses` uses SWE-bench's status names so it can go straight into its grading.
    `valid` is False when the log cannot be trusted (timeout, `test_patch` failed to
    apply, the suite never ran); `statuses` is then empty and the task scores unresolved.
    """

    __test__ = False  # not a pytest test class, despite the name

    statuses: dict[str, TestStatus]
    valid: bool
    raw_output: str  # the end of the log, truncated to MAX_OUTPUT_CHARS
    runtime_s: float
    timed_out: bool


def run_tests(
    sandbox: Sandbox, task: Task, timeout: float = DEFAULT_TEST_TIMEOUT_S
) -> TestResult:
    """Run the official SWE-bench eval script for `task` and parse its log.

    The script resets the test files to `base_commit`, applies `test_patch`, reinstalls
    the package where the repo needs it, and runs this repo's test command. Call it only
    after the candidate patch is in place, since it reveals the hidden tests.
    """
    test_spec = make_test_spec(task.model_dump())
    sandbox.write_file(_EVAL_SCRIPT_PATH, test_spec.eval_script)
    result = sandbox.exec(f"bash {_EVAL_SCRIPT_PATH}", timeout=timeout)

    statuses: dict[str, Any] = {}
    valid = False
    if not result.timed_out:
        statuses, valid = _parse_log(test_spec, result.output)
    return TestResult(
        statuses=statuses,
        valid=valid,
        raw_output=result.output[-MAX_OUTPUT_CHARS:],
        runtime_s=result.runtime_s,
        timed_out=result.timed_out,
    )


def _parse_log(test_spec: Any, log: str) -> tuple[dict[str, Any], bool]:
    # get_logs_eval reads from a file and runs SWE-bench's validity checks.
    with tempfile.NamedTemporaryFile("w", suffix=".log") as f:
        f.write(log)
        f.flush()
        statuses, valid = get_logs_eval(test_spec, f.name)
    return statuses, bool(valid)
