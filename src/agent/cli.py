"""The `harness` command: evaluate patches, and run the agent on tasks.

    uv run harness solve psf__requests-3362
    uv run harness solve --dev
    uv run harness solve --dev --model anthropic:claude-sonnet-5-5
    uv run harness run pallets__flask-4992 --gold
    uv run harness validate

`solve` runs the agent on each task, prints a line per task as it finishes and a
table at the end, and saves every run under `runs/<run_id>/`. It exits 0 if every
task is resolved and 1 if not. `run` and `validate` come from the harness.
"""

from __future__ import annotations

import argparse
import sys

from agent.config import DEFAULT_MODEL, AgentConfig
from agent.harness import cli as harness_cli
from agent.harness.dataset import Task, load_dev_tasks, load_task
from agent.runs import RUNS_DIR, Run, new_run_id
from agent.solve import solve


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run a subcommand; return the process exit code."""
    parser = argparse.ArgumentParser(prog="harness", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    harness_cli.add_commands(sub)

    defaults = AgentConfig()
    solve_cmd = sub.add_parser(
        "solve", help="run the agent on tasks and score its diffs"
    )
    solve_cmd.add_argument("instance_ids", nargs="*", metavar="instance_id")
    solve_cmd.add_argument("--dev", action="store_true", help="every dev task")
    solve_cmd.add_argument("--model", default=DEFAULT_MODEL, help="'provider:model'")
    solve_cmd.add_argument("--max-turns", type=int, default=defaults.max_turns)
    solve_cmd.add_argument("--run-id", help="name for this batch (default: UTC time)")

    args = parser.parse_args(argv)
    if args.command != "solve":
        return harness_cli.dispatch(args)
    if args.dev == bool(args.instance_ids):
        parser.error("solve takes instance ids or --dev, not both")
    selected = (
        load_dev_tasks() if args.dev else [load_task(i) for i in args.instance_ids]
    )
    config = AgentConfig(model=args.model, max_turns=args.max_turns)
    return _solve(selected, config, args.run_id or new_run_id())


def _solve(tasks: list[Task], config: AgentConfig, run_id: str) -> int:
    print(f"run {run_id}: {len(tasks)} task(s) with {config.model}", flush=True)
    runs: list[Run] = []
    crashed: list[str] = []
    for task in tasks:
        # One task's crash (Docker, a bug in the harness) must not lose the batch.
        try:
            run = solve(task, config, run_id=run_id)
        except Exception as e:
            crashed.append(task.instance_id)
            print(
                f"{task.instance_id:28} CRASHED       {type(e).__name__}: {e}",
                flush=True,
            )
            continue
        runs.append(run)
        print(_row(run), flush=True)

    resolved = sum(run.resolved for run in runs)
    print(f"\n{resolved}/{len(tasks)} resolved", end="")
    if crashed:
        print(f", {len(crashed)} crashed", end="")
    if runs:
        cost = sum(run.cost_usd for run in runs)
        runtime = sum(run.runtime_s for run in runs)
        print(
            f"  (cost ${cost:.4f}, ${cost / len(runs):.4f}/task;"
            f" runtime {runtime:.0f}s, {runtime / len(runs):.0f}s/task)",
            end="",
        )
    print(f"\nruns saved in {RUNS_DIR / run_id}")
    return 0 if resolved == len(tasks) else 1


def _row(run: Run) -> str:
    verdict = "RESOLVED" if run.resolved else "NOT RESOLVED"
    tokens = (run.input_tokens + run.output_tokens) / 1000
    errors = sum(call.is_error for call in run.tool_calls)
    return (
        f"{run.instance_id:28} {verdict:13} {run.stop_reason:14}"
        f" {run.turns:3} turns {tokens:6.0f}k tok  ${run.cost_usd:.4f}"
        f"  {errors:2} tool errors  {run.runtime_s:4.0f}s"
    )


if __name__ == "__main__":
    sys.exit(main())
