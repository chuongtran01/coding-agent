"""The `harness` command: evaluate patches, and run the agent on tasks.

    uv run harness solve psf__requests-3362
    uv run harness solve --dev
    uv run harness solve --dev --model anthropic:claude-sonnet-5-5
    uv run harness solve --dev --context none,full_file,repo_map --repeat 3 --run-id cmp
    uv run harness rank --dev
    uv run harness rank --tuning --top 10
    uv run harness run pallets__flask-4992 --gold
    uv run harness validate

`solve` runs the agent on each task, prints a line per task as it finishes and a
summary at the end, and saves every run under `runs/<run_id>/`. It exits 0 if every
run is resolved and 1 if not. `rank` ranks files from each issue, as the context
strategies do, and reports how many of the gold patch's files rank in the top k;
it makes no API calls. `run` and `validate` come from the harness.

With several context strategies or `--repeat N`, `solve` runs N rounds, and in
each round every task under every strategy, one task at a time, so drift over a
long comparison (provider load, rate limits) is spread over all strategies. The
strategy that goes first rotates from round to round. Each round and strategy is
its own batch, `<run_id>-<round>-<strategy>`, and the summary has a line per
strategy.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import NamedTuple

from agent.config import DEFAULT_MODEL, AgentConfig
from agent.context import (
    STRATEGIES,
    RepoView,
    UnknownStrategyError,
    changed_files,
    get_strategy,
    rank_files,
    recall,
)
from agent.harness import cli as harness_cli
from agent.harness.dataset import Task, load_dev_tasks, load_task, load_tuning_tasks
from agent.harness.sandbox import Sandbox
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
    solve_cmd.add_argument(
        "--context",
        type=_context_names,
        default=[defaults.context],
        help=f"context strategies, comma-separated ({', '.join(STRATEGIES)})",
    )
    solve_cmd.add_argument(
        "--repeat", type=_positive, default=1, help="rounds over every task"
    )

    rank_cmd = sub.add_parser(
        "rank", help="rank files from each issue and check the gold patch's files"
    )
    rank_cmd.add_argument("instance_ids", nargs="*", metavar="instance_id")
    rank_cmd.add_argument("--dev", action="store_true", help="every dev task")
    rank_cmd.add_argument(
        "--tuning", action="store_true", help="every ranking tuning task"
    )
    rank_cmd.add_argument(
        "--top", type=_positive, default=5, help="ranked files to list per task"
    )

    args = parser.parse_args(argv)
    if args.command not in ("solve", "rank"):
        return harness_cli.dispatch(args)
    tuning = getattr(args, "tuning", False)
    if args.dev + tuning + bool(args.instance_ids) != 1:
        parser.error(f"{args.command} takes one of: instance ids, --dev, --tuning")
    if args.dev:
        selected = load_dev_tasks()
    elif tuning:
        selected = load_tuning_tasks()
    else:
        selected = [load_task(i) for i in args.instance_ids]
    if args.command == "rank":
        return _rank(selected, args.top)
    config = AgentConfig(model=args.model, max_turns=args.max_turns)
    run_id = args.run_id or new_run_id()
    return _solve(plan(selected, args.context, args.repeat, run_id), config, run_id)


class Planned(NamedTuple):
    """One run to do: a task, the context strategy to use, and its batch."""

    task: Task
    context: str
    run_id: str


def plan(
    tasks: list[Task], contexts: list[str], repeat: int, run_id: str
) -> list[Planned]:
    """Return every run to do, in order: rounds of every task under every strategy.

    One strategy and one round is a single batch named `run_id`, as before.
    """
    if len(contexts) == 1 and repeat == 1:
        return [Planned(task, contexts[0], run_id) for task in tasks]
    planned = []
    for round_ in range(repeat):
        first = round_ % len(contexts)
        order = contexts[first:] + contexts[:first]
        for task in tasks:
            planned += [Planned(task, c, f"{run_id}-{round_ + 1}-{c}") for c in order]
    return planned


def _solve(planned: list[Planned], config: AgentConfig, run_id: str) -> int:
    contexts = list(dict.fromkeys(p.context for p in planned))
    labelled = len(contexts) > 1
    print(
        f"run {run_id}: {len(planned)} run(s) with {config.model},"
        f" context {', '.join(contexts)}",
        flush=True,
    )
    runs: list[Run] = []
    crashed: list[Planned] = []
    for p in planned:
        # One task's crash (Docker, a bug in the harness) must not lose the batch.
        try:
            run = solve(
                p.task,
                config.model_copy(update={"context": p.context}),
                run_id=p.run_id,
            )
        except Exception as e:
            crashed.append(p)
            label = f"{p.context:10} " if labelled else ""
            print(
                f"{label}{p.task.instance_id:28} CRASHED       {type(e).__name__}: {e}",
                flush=True,
            )
            continue
        runs.append(run)
        print(_row(run, labelled), flush=True)

    print()
    for context in contexts:
        print(
            _summary(
                [r for r in runs if r.config.context == context],
                attempted=sum(p.context == context for p in planned),
                crashed=sum(p.context == context for p in crashed),
                label=f"{context:10} " if labelled else "",
            )
        )
    batches = list(dict.fromkeys(p.run_id for p in planned))
    saved = RUNS_DIR / batches[0] if len(batches) == 1 else f"{RUNS_DIR / run_id}-*"
    print(f"runs saved in {saved}")
    for run in runs:
        if not run.flagged:
            continue
        print(f"\nFLAGGED {run.instance_id} ({run.run_id}):")
        for line in run.hidden_text_seen:
            print(f"  saw hidden text: {line}")
        for path in run.package_paths_outside_testbed:
            print(f"  touched its package outside /testbed: {path}")
    return 0 if sum(run.resolved for run in runs) == len(planned) else 1


def _summary(runs: list[Run], *, attempted: int, crashed: int, label: str) -> str:
    resolved = sum(run.resolved for run in runs)
    flagged = sum(run.flagged for run in runs)
    line = f"{label}{resolved}/{attempted} resolved"
    if crashed:
        line += f", {crashed} crashed"
    if flagged:
        line += f", {flagged} flagged (scores not trusted)"
    if runs:
        cost = sum(run.cost_usd for run in runs)
        runtime = sum(run.runtime_s for run in runs)
        line += (
            f"  (cost ${cost:.4f}, ${cost / len(runs):.4f}/task;"
            f" runtime {runtime:.0f}s, {runtime / len(runs):.0f}s/task)"
        )
    return line


RECALL_AT = (1, 3, 5, 10)


def _rank(tasks: list[Task], top: int) -> int:
    recalls: dict[int, list[float]] = {k: [] for k in RECALL_AT}
    for task in tasks:
        # The gold patch only grades the ranking; rank_files never sees it.
        gold = changed_files(task.gold_patch)
        start = time.monotonic()
        try:
            with Sandbox.start(task) as sandbox:
                ranked = rank_files(task.for_agent(), RepoView(sandbox))
        except Exception as e:
            print(f"{task.instance_id}  CRASHED  {type(e).__name__}: {e}", flush=True)
            continue
        paths = [f.path for f in ranked]
        found = ", ".join(
            f"{g} (#{paths.index(g) + 1})" if g in paths else f"{g} (not ranked)"
            for g in gold
        )
        print(
            f"{task.instance_id}  {len(paths)} files ranked"
            f" in {time.monotonic() - start:.0f}s; gold: {found}"
        )
        for i, f in enumerate(ranked[:top], 1):
            mark = "*" if f.path in gold else " "
            print(f"  {mark}{i:2} {f.score:6.2f}  {f.path}  [{'; '.join(f.reasons)}]")
        for k in RECALL_AT:
            value = recall(paths[:k], gold)
            if value is not None:
                recalls[k].append(value)
        print(flush=True)
    n = len(recalls[RECALL_AT[0]])
    if n:
        line = "  ".join(f"@{k} {sum(v) / n:.2f}" for k, v in recalls.items())
        print(f"ranking recall over {n} task(s):  {line}")
    return 0


def _context_names(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",")]
    for name in names:
        try:
            get_strategy(name)
        except UnknownStrategyError as e:
            raise argparse.ArgumentTypeError(str(e)) from None
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(f"a strategy is listed twice: {value!r}")
    return names


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def _row(run: Run, labelled: bool) -> str:
    verdict = "RESOLVED" if run.resolved else "NOT RESOLVED"
    tokens = (run.input_tokens + run.output_tokens) / 1000
    errors = sum(call.is_error for call in run.tool_calls)
    label = f"{run.config.context:10} " if labelled else ""
    return (
        f"{label}{run.instance_id:28} {verdict:13} {run.stop_reason:14}"
        f" {run.turns:3} turns {tokens:6.0f}k tok  ${run.cost_usd:.4f}"
        f"  {errors:2} tool errors  {run.runtime_s:4.0f}s"
        + ("  FLAGGED" if run.flagged else "")
    )


if __name__ == "__main__":
    sys.exit(main())
