"""Needs Docker and the dev task's image. Uses a scripted model, so no API key."""

from pathlib import Path

from agent.config import AgentConfig
from agent.harness.dataset import load_task
from agent.runs import load_run
from agent.solve import solve

from ..unit_tests.fakes import ScriptedModel, ai, call

TASK_ID = "pallets__flask-4992"


def test_solve_with_the_gold_patch_resolves(tmp_path: Path) -> None:
    # A stand-in agent that applies the known fix through the tools, so the run
    # exercises diff capture in the offline sandbox and scoring in a fresh one.
    task = load_task(TASK_ID)
    model = ScriptedModel(
        replies=[
            ai(call("write", "c1", path="/tmp/fix.diff", content=task.gold_patch)),
            ai(
                call(
                    "bash", "c2", command="git apply /tmp/fix.diff && rm /tmp/fix.diff"
                )
            ),
            ai(call("submit", "c3")),
        ],
        seen=[],
    )
    config = AgentConfig(model="anthropic:claude-sonnet-5-5", context="full_file")
    run = solve(task, config, run_id="it", model=model, runs_dir=tmp_path)

    assert run.stop_reason == "submitted"
    assert run.evaluation.apply.applied, run.evaluation.apply.error
    assert run.resolved
    # A real context is recorded and scored, and is not mistaken for a leak.
    assert run.context.items and run.context.chars <= config.context_budget_chars
    assert run.context_recall == 1.0  # the gold file, src/flask/config.py
    assert not run.flagged, run.hidden_text_seen
    assert load_run("it", TASK_ID, tmp_path) == run
