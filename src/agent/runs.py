"""Save and load agent runs under `runs/<run_id>/<instance_id>/`.

Each run directory holds:

    run.json         the summary: config, stop reason, turns, tokens, cost, runtimes, evaluation
    patch.diff       the diff the agent produced, exactly as it was scored
    trajectory.json  every message and tool call, for reading failed runs by hand

A run id names one batch (one invocation of `solve` over one or more tasks), so the
tasks of a batch sit side by side.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AnyMessage
from pydantic import BaseModel

from agent.config import AgentConfig
from agent.graph import StopReason, ToolCall
from agent.harness.evaluate import Evaluation

RUNS_DIR = Path("runs")

_SUMMARY = "run.json"
_DIFF = "patch.diff"
_TRAJECTORY = "trajectory.json"
_TRAJECTORY_FIELDS = {"messages", "tool_calls"}


class Run(BaseModel):
    """One attempt at one task: what the agent did, the diff and its score."""

    run_id: str
    instance_id: str
    config: AgentConfig  # includes the model string
    started_at: datetime
    stop_reason: StopReason
    error: str | None
    turns: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    diff: str
    evaluation: Evaluation
    agent_runtime_s: float  # the agent loop alone
    runtime_s: float  # the whole solve: sandbox start to the end of evaluation
    messages: list[AnyMessage]
    tool_calls: list[ToolCall]

    @property
    def resolved(self) -> bool:
        """Whether the hidden tests passed."""
        return self.evaluation.resolved


def new_run_id() -> str:
    """Return a run id from the current UTC time, e.g. `20261011-153012`."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def run_dir(run_id: str, instance_id: str, root: Path = RUNS_DIR) -> Path:
    """Return the directory that holds one run."""
    return root / run_id / instance_id


def save_run(run: Run, root: Path = RUNS_DIR) -> Path:
    """Write `run` to its directory and return the directory."""
    directory = run_dir(run.run_id, run.instance_id, root)
    directory.mkdir(parents=True, exist_ok=True)
    summary = run.model_dump_json(exclude=_TRAJECTORY_FIELDS | {"diff"}, indent=2)
    (directory / _SUMMARY).write_text(summary + "\n")
    (directory / _DIFF).write_text(run.diff)
    trajectory = run.model_dump_json(include=_TRAJECTORY_FIELDS, indent=2)
    (directory / _TRAJECTORY).write_text(trajectory + "\n")
    return directory


def load_run(run_id: str, instance_id: str, root: Path = RUNS_DIR) -> Run:
    """Read back a run written by `save_run`."""
    directory = run_dir(run_id, instance_id, root)
    data = json.loads((directory / _SUMMARY).read_text())
    data |= json.loads((directory / _TRAJECTORY).read_text())
    data["diff"] = (directory / _DIFF).read_text()
    return Run.model_validate(data)
