"""Load SWE-bench Lite rows into typed `Task` objects."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The SWE-bench team's copy of Lite. Same tasks as princeton-nlp/SWE-bench_Lite, plus the
# official image, eval script and log parser for each one.
DATASET_NAME = "SWE-bench/SWE-bench_Lite"
DEFAULT_SPLIT = "test"

# Dev tasks for building and validating the harness; see docs/DEV_TASKS.md.
# These are tuned against, so exclude them from Phase 5 benchmark numbers.
DEV_TASK_IDS: tuple[str, ...] = (
    "psf__requests-3362",
    "pytest-dev__pytest-7432",
    "pytest-dev__pytest-5227",
    "pytest-dev__pytest-7220",
    "mwaskom__seaborn-3010",
)

# Tasks for tuning the context ranking (Phase 3), drawn by repo with a fixed seed so
# that django and sympy, which the dev tasks lack, are covered. Ranking rules were
# changed after seeing them, so like the dev tasks they are excluded from Phase 5
# benchmark numbers. Only the ranking is tuned on them, never the agent.
TUNING_TASK_IDS: tuple[str, ...] = (
    "astropy__astropy-14182",
    "django__django-13590",
    "django__django-14382",
    "django__django-14672",
    "django__django-15347",
    "django__django-15498",
    "django__django-15814",
    "matplotlib__matplotlib-23964",
    "matplotlib__matplotlib-25332",
    "mwaskom__seaborn-3190",
    "pallets__flask-4045",
    "psf__requests-1963",
    "pydata__xarray-5131",
    "pylint-dev__pylint-7228",
    "pytest-dev__pytest-5221",
    "pytest-dev__pytest-7168",
    "scikit-learn__scikit-learn-10508",
    "scikit-learn__scikit-learn-13439",
    "sphinx-doc__sphinx-7975",
    "sphinx-doc__sphinx-8595",
    "sympy__sympy-11400",
    "sympy__sympy-12481",
    "sympy__sympy-18532",
    "sympy__sympy-21171",
    "sympy__sympy-24066",
)


class AgentTask(BaseModel):
    """The part of a task the agent may see: no patches, no tests, no eval script."""

    model_config = ConfigDict(frozen=True)

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str


class Task(BaseModel):
    """One SWE-bench Lite instance.

    `gold_patch`, `test_patch` and `eval_script` (which embeds `test_patch`) are
    evaluation-only and must never reach the agent.
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
    version: str
    image: str  # published Docker image with the repo installed at /testbed
    eval_type: str
    log_parser: str  # name of the SWE-bench parser for this repo's test output
    eval_script: str  # official script: apply test_patch, run the tests

    @field_validator("FAIL_TO_PASS", "PASS_TO_PASS", mode="before")
    @classmethod
    def _parse_json_list(cls, value: Any) -> Any:
        """Parse a test list the dataset stores as a JSON string."""
        return json.loads(value) if isinstance(value, str) else value

    def for_agent(self) -> AgentTask:
        """Return the view of this task that is safe to hand to the agent."""
        return AgentTask(
            instance_id=self.instance_id,
            repo=self.repo,
            base_commit=self.base_commit,
            problem_statement=self.problem_statement,
        )


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


def load_tuning_tasks() -> list[Task]:
    """Load the ranking tuning tasks listed in `TUNING_TASK_IDS`."""
    tasks = load_tasks()
    return [tasks[i] for i in TUNING_TASK_IDS]
