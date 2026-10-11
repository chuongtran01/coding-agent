"""Command line for the harness.

    uv run harness run pallets__flask-4992 --gold
    uv run harness run pallets__flask-4992 --patch fix.diff
    uv run harness validate

`run` exits 0 if the task is resolved and 1 if not. `validate` runs the gold, empty
and a broken patch on every dev task and exits 1 if any result is unexpected.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from agent.harness.dataset import load_dev_tasks, load_task
from agent.harness.evaluate import Evaluation, evaluate
from agent.harness.runner import DEFAULT_TEST_TIMEOUT_S

_MAX_LISTED_FAILURES = 10
_OUTPUT_TAIL_LINES = 20


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run a subcommand; return the process exit code."""
    parser = argparse.ArgumentParser(prog="harness", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="evaluate one patch on one task")
    run.add_argument("instance_id")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--patch", type=Path, help="diff file to evaluate ('-' = stdin)"
    )
    source.add_argument("--gold", action="store_true", help="use the gold patch")
    source.add_argument("--empty", action="store_true", help="use an empty patch")
    run.add_argument("--timeout", type=float, default=DEFAULT_TEST_TIMEOUT_S)
    run.add_argument("-v", "--verbose", action="store_true", help="show test output")

    validate = sub.add_parser("validate", help="check gold/empty/broken on dev tasks")
    validate.add_argument("--repeat", type=int, default=2, help="gold runs per task")

    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    return _validate(args.repeat)


def _run(args: argparse.Namespace) -> int:
    task = load_task(args.instance_id)
    if args.gold:
        diff = task.gold_patch
    elif args.empty:
        diff = ""
    elif str(args.patch) == "-":
        diff = sys.stdin.read()
    else:
        diff = args.patch.read_text()

    result = evaluate(task, diff, test_timeout=args.timeout)
    print(_report(result, verbose=args.verbose))
    return 0 if result.resolved else 1


def _validate(repeat: int) -> int:
    rows: list[tuple[str, str, bool, Evaluation]] = []
    for task in load_dev_tasks():
        cases = [("gold", task.gold_patch, True)] * repeat + [
            ("empty", "", False),
            ("broken", break_patch(task.gold_patch), False),
        ]
        for label, diff, expect_resolved in cases:
            result = evaluate(task, diff)
            ok = result.resolved == expect_resolved
            if label == "broken":
                ok = ok and not result.apply.applied and bool(result.apply.error)
            rows.append((label, task.instance_id, ok, result))
            print(
                f"{'ok  ' if ok else 'FAIL'} {label:6} {_summary(result)}", flush=True
            )

    failures = [r for r in rows if not r[2]]
    print(f"\n{len(rows) - len(failures)}/{len(rows)} runs as expected")
    for label, _, _, result in failures:
        print(f"\n--- unexpected: {label} ---")
        print(_report(result, verbose=True))
    return 1 if failures else 0


def break_patch(diff: str) -> str:
    """Return `diff` with one removed line altered, so its hunk no longer applies."""
    return re.sub(
        r"^-(?!--)(.*)$", r"-\1  # not in the file", diff, count=1, flags=re.M
    )


def _summary(result: Evaluation) -> str:
    verdict = "RESOLVED" if result.resolved else "NOT RESOLVED"
    line = f"{result.instance_id:28} {verdict:13}"
    if result.score is not None:
        f2p, p2p = result.score.fail_to_pass, result.score.pass_to_pass
        line += (
            f" F2P {len(f2p.success)}/{len(f2p.success) + len(f2p.failure)}"
            f"  P2P {len(p2p.success)}/{len(p2p.success) + len(p2p.failure)}"
        )
    elif not result.apply.applied:
        line += " patch did not apply"
    tests_s = f"tests {result.tests.runtime_s:.1f}s, " if result.tests else ""
    return f"{line}  ({tests_s}total {result.runtime_s:.1f}s)"


def _report(result: Evaluation, verbose: bool) -> str:
    lines = [_summary(result)]
    if not result.apply.applied:
        lines += ["", "git apply error:", _indent(result.apply.error or "")]
    elif result.tests is not None and not result.tests.valid:
        reason = "timed out" if result.tests.timed_out else "log not usable"
        lines += ["", f"test run invalid ({reason}); end of output:"]
        lines += [_indent(_tail(result.tests.raw_output))]
    elif result.score is not None and not result.resolved:
        for name, outcomes in [
            ("FAIL_TO_PASS", result.score.fail_to_pass),
            ("PASS_TO_PASS", result.score.pass_to_pass),
        ]:
            if outcomes.failure:
                lines += ["", f"{name} failing ({len(outcomes.failure)}):"]
                lines += [f"  {t}" for t in outcomes.failure[:_MAX_LISTED_FAILURES]]
                if len(outcomes.failure) > _MAX_LISTED_FAILURES:
                    lines.append(
                        f"  ... {len(outcomes.failure) - _MAX_LISTED_FAILURES} more"
                    )
    if verbose and result.tests is not None:
        lines += ["", "end of test output:", _indent(_tail(result.tests.raw_output))]
    return "\n".join(lines)


def _tail(text: str) -> str:
    return "\n".join(text.rstrip().splitlines()[-_OUTPUT_TAIL_LINES:])


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())


if __name__ == "__main__":
    sys.exit(main())
