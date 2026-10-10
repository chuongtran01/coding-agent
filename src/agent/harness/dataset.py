"""Load SWE-bench Lite rows into typed `Task` objects."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DATASET_NAME = "princeton-nlp/SWE-bench_Lite"
DEFAULT_SPLIT = "test"

# Dev tasks for building and validating the harness; see docs/DEV_TASKS.md.
# These are tuned against, so exclude them from Phase 5 benchmark numbers.
DEV_TASK_IDS: tuple[str, ...] = (
    "psf__requests-3362",
    "pallets__flask-4992",
    "pytest-dev__pytest-5227",
    "pytest-dev__pytest-7220",
    "mwaskom__seaborn-3010",
)


@dataclass(frozen=True)
class Task:
    """One SWE-bench Lite instance.

    `gold_patch` and `test_patch` are evaluation-only and must never reach the agent.
    """

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    FAIL_TO_PASS: tuple[str, ...]
    PASS_TO_PASS: tuple[str, ...]
    gold_patch: str
    test_patch: str


def _parse_tests(value: str | list[str]) -> tuple[str, ...]:
    """Parse a test list stored as a JSON string (or already a list)."""
    tests = json.loads(value) if isinstance(value, str) else value
    if not isinstance(tests, list) or not all(isinstance(t, str) for t in tests):
        raise ValueError(f"expected a list of test ids, got {tests!r}")
    return tuple(tests)


def row_to_task(row: Mapping[str, Any]) -> Task:
    """Convert one dataset row into a `Task`."""
    return Task(
        instance_id=row["instance_id"],
        repo=row["repo"],
        base_commit=row["base_commit"],
        problem_statement=row["problem_statement"],
        FAIL_TO_PASS=_parse_tests(row["FAIL_TO_PASS"]),
        PASS_TO_PASS=_parse_tests(row["PASS_TO_PASS"]),
        gold_patch=row["patch"],
        test_patch=row["test_patch"],
    )


def load_tasks(split: str = DEFAULT_SPLIT) -> dict[str, Task]:
    """Load every task in a split, keyed by instance id."""
    from datasets import load_dataset  # type: ignore[import-untyped]

    ds = load_dataset(DATASET_NAME, split=split)
    return {row["instance_id"]: row_to_task(row) for row in ds}


def load_task(instance_id: str, split: str = DEFAULT_SPLIT) -> Task:
    """Load a single task by instance id."""
    tasks = load_tasks(split)
    try:
        return tasks[instance_id]
    except KeyError:
        raise KeyError(f"{instance_id!r} not in {DATASET_NAME} ({split})") from None


def load_dev_tasks() -> list[Task]:
    """Load the dev tasks listed in `DEV_TASK_IDS`."""
    tasks = load_tasks()
    return [tasks[i] for i in DEV_TASK_IDS]
