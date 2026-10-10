"""Load SWE-bench Lite rows into typed `Task` objects."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class Task(BaseModel):
    """One SWE-bench Lite instance.

    `gold_patch` and `test_patch` are evaluation-only and must never reach the agent.
    Build one from a dataset row with `Task.model_validate(row)`.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    instance_id: str
    repo: str
    base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    problem_statement: str
    FAIL_TO_PASS: tuple[str, ...]
    PASS_TO_PASS: tuple[str, ...]
    gold_patch: str = Field(validation_alias="patch")
    test_patch: str

    @field_validator("FAIL_TO_PASS", "PASS_TO_PASS", mode="before")
    @classmethod
    def _parse_json_list(cls, value: Any) -> Any:
        """Parse a test list the dataset stores as a JSON string."""
        return json.loads(value) if isinstance(value, str) else value


def load_tasks(split: str = DEFAULT_SPLIT) -> dict[str, Task]:
    """Load every task in a split, keyed by instance id."""
    from datasets import load_dataset  # type: ignore[import-untyped]

    ds = load_dataset(DATASET_NAME, split=split)
    return {row["instance_id"]: Task.model_validate(row) for row in ds}


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
