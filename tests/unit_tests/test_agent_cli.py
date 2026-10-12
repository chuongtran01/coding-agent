import argparse
from datetime import datetime, timezone

import pytest

from agent import cli
from agent.config import AgentConfig
from agent.harness import cli as harness_cli
from agent.harness.dataset import Task
from agent.harness.evaluate import Evaluation
from agent.harness.patch import ApplyResult
from agent.runs import Run


def task(instance_id: str) -> Task:
    return Task(
        instance_id=instance_id,
        repo="psf/requests",
        base_commit="a" * 40,
        problem_statement="",
        FAIL_TO_PASS=(),
        PASS_TO_PASS=(),
        gold_patch="",
        test_patch="",
        version="2.0",
        image="img",
        eval_type="pass_and_fail",
        log_parser="parse_log_pytest",
        eval_script="",
    )


DEV = [task("a__a-1"), task("b__b-2")]


def fake_run(t: Task, config: AgentConfig, run_id: str, resolved: bool) -> Run:
    return Run(
        run_id=run_id,
        instance_id=t.instance_id,
        config=config,
        started_at=datetime.now(timezone.utc),
        stop_reason="submitted",
        error=None,
        turns=4,
        input_tokens=9000,
        output_tokens=1000,
        cost_usd=0.01,
        diff="",
        evaluation=Evaluation(
            instance_id=t.instance_id,
            resolved=resolved,
            apply=ApplyResult(applied=True),
            tests=None,
            score=None,
            runtime_s=1.0,
        ),
        agent_runtime_s=5.0,
        runtime_s=10.0,
        messages=[],
        tool_calls=[],
    )


@pytest.fixture
def solved(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, AgentConfig, str]]:
    """Stub out the dataset and solve(); 'b__b-2' is unresolved, 'boom' crashes."""
    calls: list[tuple[str, AgentConfig, str]] = []

    def fake_solve(t: Task, config: AgentConfig, *, run_id: str) -> Run:
        calls.append((t.instance_id, config, run_id))
        if t.instance_id == "boom":
            raise RuntimeError("docker went away")
        run = fake_run(t, config, run_id, resolved=t.instance_id != "b__b-2")
        if t.instance_id == "leaky":
            run.package_paths_outside_testbed = ["/opt/x/site-packages/leaky/a.py"]
        return run

    monkeypatch.setattr(cli, "solve", fake_solve)
    monkeypatch.setattr(cli, "load_dev_tasks", lambda: DEV)
    monkeypatch.setattr(cli, "load_task", task)
    return calls


def test_solve_dev_runs_every_dev_task_in_one_batch(
    solved: list[tuple[str, AgentConfig, str]], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["solve", "--dev", "--run-id", "batch1"]) == 1  # one unresolved
    assert [(i, r) for i, _, r in solved] == [
        ("a__a-1", "batch1"),
        ("b__b-2", "batch1"),
    ]
    out = capsys.readouterr().out
    assert "a__a-1" in out and "RESOLVED" in out and "NOT RESOLVED" in out
    assert "1/2 resolved" in out and "$0.0100/task" in out
    assert "runs/batch1" in out


def test_solve_ids_with_model_and_turns(
    solved: list[tuple[str, AgentConfig, str]],
) -> None:
    argv = [
        "solve",
        "a__a-1",
        "--model",
        "anthropic:claude-sonnet-5-5",
        "--max-turns",
        "7",
    ]
    assert cli.main(argv) == 0
    ((instance_id, config, _),) = solved
    assert instance_id == "a__a-1"
    assert config.model == "anthropic:claude-sonnet-5-5" and config.max_turns == 7


def test_a_crash_does_not_stop_the_batch(
    solved: list[tuple[str, AgentConfig, str]], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["solve", "boom", "a__a-1"]) == 1
    assert [i for i, _, _ in solved] == ["boom", "a__a-1"]
    out = capsys.readouterr().out
    assert "boom" in out and "CRASHED" in out and "docker went away" in out
    assert "1/2 resolved, 1 crashed" in out


@pytest.mark.parametrize("argv", [["solve"], ["solve", "a__a-1", "--dev"]])
def test_solve_needs_ids_or_dev(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(argv)


def test_harness_commands_still_work(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_dispatch(args: argparse.Namespace) -> int:
        seen.append(args.command)
        return 0

    monkeypatch.setattr(harness_cli, "dispatch", fake_dispatch)
    assert cli.main(["validate"]) == 0
    assert seen == ["validate"]


def test_flagged_runs_are_reported(
    solved: list[tuple[str, AgentConfig, str]], capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["solve", "leaky", "a__a-1"])
    out = capsys.readouterr().out
    assert "FLAGGED" in out and "1 flagged (scores not trusted)" in out
    assert (
        "touched its package outside /testbed: /opt/x/site-packages/leaky/a.py" in out
    )
