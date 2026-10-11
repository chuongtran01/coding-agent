"""Solve one task end to end: run the agent, capture its diff, score it, save the run."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from agent.config import AgentConfig
from agent.graph import run_agent
from agent.harness.dataset import Task
from agent.harness.evaluate import evaluate
from agent.harness.sandbox import Sandbox, SandboxError
from agent.model import make_model
from agent.runs import RUNS_DIR, Run, new_run_id, save_run
from agent.tools import Shell, Toolbox

DIFF_TIMEOUT_S = 120
_DIFF_LOG = "/tmp/.agent-diff.log"


def solve(
    task: Task,
    config: AgentConfig,
    *,
    run_id: str | None = None,
    model: BaseChatModel | None = None,
    runs_dir: Path | None = RUNS_DIR,
) -> Run:
    """Run the agent on `task`, score its diff, and save the run unless `runs_dir` is None.

    The agent works in an offline sandbox and sees only `task.for_agent()`; the diff is
    captured there, then scored by `evaluate()` in a fresh container. Hitting a limit
    still ends in a scored diff. `model` overrides `make_model(config)`, for tests.
    """
    run_id = run_id or new_run_id()
    started_at = datetime.now(timezone.utc)
    start = time.monotonic()
    chat_model = model or make_model(config)
    with Sandbox.start(task) as sandbox:
        toolbox = Toolbox(sandbox, bash_timeout_s=config.bash_timeout_s)
        result = run_agent(task.for_agent(), chat_model, toolbox, config)
        diff = capture_diff(sandbox)
    evaluation = evaluate(task, diff)
    run = Run(
        run_id=run_id,
        instance_id=task.instance_id,
        config=config,
        started_at=started_at,
        stop_reason=result.stop_reason,
        error=result.error,
        turns=result.turns,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        diff=diff,
        evaluation=evaluation,
        agent_runtime_s=result.runtime_s,
        runtime_s=time.monotonic() - start,
        messages=result.messages,
        tool_calls=result.tool_calls,
    )
    if runs_dir is not None:
        save_run(run, runs_dir)
    return run


def capture_diff(shell: Shell) -> str:
    """Return every change in the repo against HEAD, new files included, as a diff."""
    # Git's own messages (warnings from `git add`, errors) go to a log, so that on
    # success the output is the diff alone.
    result = shell.exec(
        f"git add -A > {_DIFF_LOG} 2>&1 && git diff --cached HEAD 2>> {_DIFF_LOG} "
        f"|| {{ cat {_DIFF_LOG}; exit 1; }}",
        timeout=DIFF_TIMEOUT_S,
    )
    if result.exit_code != 0:
        raise SandboxError(f"failed to capture the diff:\n{result.output}")
    return result.output
