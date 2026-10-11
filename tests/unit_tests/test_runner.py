from typing import cast

from agent.harness.dataset import Task
from agent.harness.runner import MAX_OUTPUT_CHARS, run_tests
from agent.harness.sandbox import ExecResult, Sandbox

TASK = Task(
    instance_id="psf__requests-0000",
    repo="psf/requests",
    base_commit="a" * 40,
    problem_statement="",
    FAIL_TO_PASS=("test_x.py::test_new",),
    PASS_TO_PASS=("test_x.py::test_a",),
    gold_patch="",
    test_patch="",
    version="2.0",
    image="img",
    eval_type="pass_and_fail",
    log_parser="parse_log_pytest",
    eval_script=(
        "#!/bin/bash\nset -uxo pipefail\n"
        ": '>>>>> Start Test Output'\npytest -rA test_x.py\n"
        ": '>>>>> End Test Output'\n"
    ),
)

LOG = """+ : '>>>>> Start Test Output'
+ pytest -rA test_x.py
PASSED test_x.py::test_a
FAILED test_x.py::test_new - AssertionError
+ SWEBENCH_TEST_EXIT_CODE=1
+ : '>>>>> End Test Output'
>>>>> Test Exit Code: 1
"""


class FakeSandbox:
    def __init__(self, output: str, timed_out: bool = False) -> None:
        self.output = output
        self.timed_out = timed_out
        self.files: dict[str, str] = {}

    def write_file(self, path: str, content: str) -> None:
        self.files[path] = content

    def exec(self, cmd: str, timeout: float = 300) -> ExecResult:
        return ExecResult(
            exit_code=124 if self.timed_out else 0,
            output=self.output,
            runtime_s=2.5,
            timed_out=self.timed_out,
        )


def test_parses_statuses_with_swebench_parser() -> None:
    fake = FakeSandbox(LOG)
    result = run_tests(cast(Sandbox, fake), TASK)
    assert result.valid
    assert result.statuses == {
        "test_x.py::test_a": "PASSED",
        "test_x.py::test_new": "FAILED",
    }
    assert result.runtime_s == 2.5
    [script] = fake.files.values()
    assert "SWEBENCH_TEST_EXIT_CODE=$?" in script  # exit code recorded for grading


def test_timeout_is_invalid() -> None:
    result = run_tests(cast(Sandbox, FakeSandbox(LOG, timed_out=True)), TASK)
    assert result.timed_out
    assert not result.valid
    assert result.statuses == {}


def test_missing_markers_is_invalid() -> None:
    result = run_tests(cast(Sandbox, FakeSandbox("error: patch failed\n")), TASK)
    assert not result.valid
    assert result.statuses == {}


def test_raw_output_keeps_the_end() -> None:
    log = "x" * MAX_OUTPUT_CHARS + LOG
    result = run_tests(cast(Sandbox, FakeSandbox(log)), TASK)
    assert len(result.raw_output) == MAX_OUTPUT_CHARS
    assert result.raw_output.endswith(LOG)
