from pathlib import Path

import pytest

from agent.harness import cli
from agent.harness.dataset import Task
from agent.harness.evaluate import Evaluation
from agent.harness.patch import ApplyResult
from agent.harness.runner import TestResult
from agent.harness.scoring import Score, TestOutcomes

GOLD = """diff --git a/x.py b/x.py
--- a/x.py
+++ b/x.py
@@ -1,2 +1,2 @@
 keep
-old
+new
"""

TASK = Task(
    instance_id="psf__requests-0000",
    repo="psf/requests",
    base_commit="a" * 40,
    problem_statement="",
    FAIL_TO_PASS=("f1",),
    PASS_TO_PASS=("p1",),
    gold_patch=GOLD,
    test_patch="",
    version="2.0",
    image="img",
    eval_type="pass_and_fail",
    log_parser="parse_log_pytest",
    eval_script="",
)


def _evaluation(resolved: bool, diff: str) -> Evaluation:
    if diff == "broken":
        return Evaluation(
            instance_id=TASK.instance_id,
            resolved=False,
            apply=ApplyResult(applied=False, error="error: patch failed: x.py:1"),
            tests=None,
            score=None,
            runtime_s=1.0,
        )
    return Evaluation(
        instance_id=TASK.instance_id,
        resolved=resolved,
        apply=ApplyResult(applied=True),
        tests=TestResult(
            statuses={}, valid=True, raw_output="", runtime_s=2.0, timed_out=False
        ),
        score=Score(
            resolved=resolved,
            status="RESOLVED_FULL" if resolved else "RESOLVED_NO",
            fail_to_pass=TestOutcomes(
                success=["f1"] if resolved else [], failure=[] if resolved else ["f1"]
            ),
            pass_to_pass=TestOutcomes(success=["p1"], failure=[]),
        ),
        runtime_s=3.0,
    )


@pytest.fixture
def evaluated(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub out the dataset and Docker; record the diffs passed to evaluate()."""
    diffs: list[str] = []

    def fake_evaluate(task: Task, diff: str, test_timeout: float = 0) -> Evaluation:
        diffs.append(diff)
        return _evaluation(resolved=diff == GOLD, diff=diff)

    monkeypatch.setattr(cli, "load_task", lambda instance_id: TASK)
    monkeypatch.setattr(cli, "evaluate", fake_evaluate)
    return diffs


def test_break_patch_changes_a_removed_line_only() -> None:
    broken = cli.break_patch(GOLD)
    assert "-old  # not in the file\n" in broken
    assert "--- a/x.py\n" in broken
    assert broken.replace("  # not in the file", "") == GOLD


def test_run_gold_exits_zero(
    evaluated: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["run", TASK.instance_id, "--gold"]) == 0
    assert evaluated == [GOLD]
    assert "RESOLVED" in capsys.readouterr().out


def test_run_empty_lists_failing_tests(
    evaluated: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["run", TASK.instance_id, "--empty"]) == 1
    assert evaluated == [""]
    out = capsys.readouterr().out
    assert "NOT RESOLVED" in out and "FAIL_TO_PASS failing (1):\n  f1" in out


def test_run_patch_file_shows_apply_error(
    evaluated: list[str], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    patch = tmp_path / "fix.diff"
    patch.write_text("broken")
    assert cli.main(["run", TASK.instance_id, "--patch", str(patch)]) == 1
    out = capsys.readouterr().out
    assert "patch did not apply" in out and "error: patch failed: x.py:1" in out


def test_run_requires_one_patch_source() -> None:
    with pytest.raises(SystemExit):
        cli.main(["run", TASK.instance_id])
