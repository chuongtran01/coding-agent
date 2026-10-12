"""The agent-side isolation and limit checks in real containers (Phase 2, task 7).

Needs Docker and the dev task images. Uses a scripted model, so no API key.
"""

import time
from pathlib import Path

from agent.config import AgentConfig
from agent.graph import run_agent
from agent.harness.dataset import load_task
from agent.harness.sandbox import Sandbox
from agent.solve import solve
from agent.tools import Toolbox

from ..unit_tests.fakes import ScriptedModel, ai, call

CONFIG = AgentConfig(model="anthropic:claude-sonnet-5-5")
# A newer requests release in conda's base environment (see docs/DEV_TASKS.md).
NEWER_REQUESTS = "/opt/miniconda3/lib/python3.11/site-packages/requests/models.py"


def test_reading_the_newer_requests_release_is_flagged(tmp_path: Path) -> None:
    task = load_task("psf__requests-3362")
    model = ScriptedModel(
        replies=[
            ai(call("read", "c1", path=NEWER_REQUESTS)),
            ai(call("bash", "c2", command="git grep -n iter_content requests/")),
            ai(call("submit", "c3")),
        ],
        seen=[],
    )
    run = solve(task, CONFIG, model=model, runs_dir=tmp_path)
    assert run.package_paths_outside_testbed == [NEWER_REQUESTS]
    assert run.flagged


def test_an_honest_run_is_not_flagged(tmp_path: Path) -> None:
    task = load_task("pallets__flask-4992")
    model = ScriptedModel(
        replies=[
            ai(
                call(
                    "bash", "c1", command="git grep -n 'def from_file' && git log -p -3"
                )
            ),
            ai(call("read", "c2", path="src/flask/config.py")),
            ai(call("submit", "c3")),
        ],
        seen=[],
    )
    run = solve(task, CONFIG, model=model, runs_dir=tmp_path)
    assert not run.flagged, (run.hidden_text_seen, run.package_paths_outside_testbed)


def test_a_hanging_command_times_out_and_the_run_goes_on() -> None:
    task = load_task("psf__requests-3362")
    config = CONFIG.model_copy(update={"bash_timeout_s": 5})
    model = ScriptedModel(
        replies=[ai(call("bash", "c1", command="sleep 600")), ai(call("submit", "c2"))],
        seen=[],
    )
    start = time.monotonic()
    with Sandbox.start(task) as sb:
        result = run_agent(
            task.for_agent(), model, Toolbox(sb, bash_timeout_s=5), config
        )
    assert time.monotonic() - start < 60
    hung = result.tool_calls[0]
    assert hung.is_error and "timed out after 5s" in hung.output
    assert result.stop_reason == "submitted"


def test_the_wall_clock_stops_a_run_stuck_in_slow_commands() -> None:
    task = load_task("psf__requests-3362")
    config = CONFIG.model_copy(update={"bash_timeout_s": 5, "max_wall_clock_s": 8})
    replies = [ai(call("bash", f"c{i}", command="sleep 600")) for i in range(10)]
    model = ScriptedModel(replies=replies, seen=[])
    start = time.monotonic()
    with Sandbox.start(task) as sb:
        result = run_agent(
            task.for_agent(), model, Toolbox(sb, bash_timeout_s=5), config
        )
    assert result.stop_reason == "max_wall_clock"
    assert result.turns == 2  # overshoot is at most one command's timeout
    assert time.monotonic() - start < 60
