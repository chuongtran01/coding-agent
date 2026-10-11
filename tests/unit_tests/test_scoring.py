from agent.harness.dataset import Task
from agent.harness.runner import TestResult, TestStatus
from agent.harness.scoring import is_resolved, score

TASK = Task(
    instance_id="psf__requests-0000",
    repo="psf/requests",
    base_commit="a" * 40,
    problem_statement="",
    FAIL_TO_PASS=("f1", "f2"),
    PASS_TO_PASS=("p1", "p2"),
    gold_patch="",
    test_patch="",
    version="2.0",
    image="img",
    eval_type="pass_and_fail",
    log_parser="parse_log_pytest",
    eval_script="",
)

ALL_PASS: dict[str, TestStatus] = {
    "f1": "PASSED",
    "f2": "PASSED",
    "p1": "PASSED",
    "p2": "PASSED",
}


def _result(statuses: dict[str, TestStatus], valid: bool = True) -> TestResult:
    return TestResult(
        statuses=statuses, valid=valid, raw_output="", runtime_s=1.0, timed_out=False
    )


def test_all_listed_tests_pass_is_resolved() -> None:
    s = score(TASK, _result({**ALL_PASS, "unlisted": "ERROR"}))
    assert s.resolved and s.status == "RESOLVED_FULL"
    assert s.fail_to_pass.failure == [] and s.pass_to_pass.failure == []


def test_some_fail_to_pass_failing_is_partial_not_resolved() -> None:
    s = score(TASK, _result({**ALL_PASS, "f2": "FAILED"}))
    assert not s.resolved and s.status == "RESOLVED_PARTIAL"
    assert s.fail_to_pass.failure == ["f2"]


def test_no_fail_to_pass_passing_is_not_resolved() -> None:
    s = score(TASK, _result({**ALL_PASS, "f1": "FAILED", "f2": "ERROR"}))
    assert s.status == "RESOLVED_NO"


def test_pass_to_pass_regression_is_not_resolved() -> None:
    s = score(TASK, _result({**ALL_PASS, "p1": "FAILED"}))
    assert not s.resolved and s.status == "RESOLVED_NO"
    assert s.pass_to_pass.failure == ["p1"]


def test_missing_listed_test_counts_as_failure() -> None:
    statuses = dict(ALL_PASS)
    del statuses["f1"]
    s = score(TASK, _result(statuses))
    assert not s.resolved
    assert s.fail_to_pass.failure == ["f1"]


def test_xfail_counts_as_pass() -> None:
    assert is_resolved(TASK, _result({**ALL_PASS, "f1": "XFAIL", "p1": "XFAIL"}))


def test_skipped_fail_to_pass_is_not_a_fix() -> None:
    assert not is_resolved(TASK, _result({**ALL_PASS, "f1": "SKIPPED"}))


def test_skipped_pass_to_pass_is_not_a_regression() -> None:
    assert is_resolved(TASK, _result({**ALL_PASS, "p1": "SKIPPED"}))


def test_invalid_run_is_never_resolved() -> None:
    s = score(TASK, _result(ALL_PASS, valid=False))
    assert not s.resolved and s.status == "RESOLVED_NO"
    assert s.fail_to_pass.failure == ["f1", "f2"]
