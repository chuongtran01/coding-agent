import json

import pytest

from agent.harness.dataset import Task, _parse_tests, row_to_task

ROW = {
    "repo": "psf/requests",
    "instance_id": "psf__requests-0000",
    "base_commit": "a" * 40,
    "patch": "diff --git a/x.py b/x.py\n",
    "test_patch": "diff --git a/test_x.py b/test_x.py\n",
    "problem_statement": "Something is broken.",
    "hints_text": "",
    "created_at": "2020-01-01T00:00:00Z",
    "version": "2.0",
    "FAIL_TO_PASS": json.dumps(["test_x.py::test_new"]),
    "PASS_TO_PASS": json.dumps(["test_x.py::test_a", "test_x.py::test_b"]),
    "environment_setup_commit": "b" * 40,
}


def test_row_to_task_maps_fields() -> None:
    task = row_to_task(ROW)
    assert task == Task(
        instance_id="psf__requests-0000",
        repo="psf/requests",
        base_commit="a" * 40,
        problem_statement="Something is broken.",
        FAIL_TO_PASS=("test_x.py::test_new",),
        PASS_TO_PASS=("test_x.py::test_a", "test_x.py::test_b"),
        gold_patch="diff --git a/x.py b/x.py\n",
        test_patch="diff --git a/test_x.py b/test_x.py\n",
    )


def test_task_is_hashable_and_frozen() -> None:
    task = row_to_task(ROW)
    hash(task)
    with pytest.raises(AttributeError):
        task.repo = "other"  # type: ignore[misc]


def test_parse_tests_accepts_list_and_empty() -> None:
    assert _parse_tests(["a", "b"]) == ("a", "b")
    assert _parse_tests("[]") == ()


def test_parse_tests_rejects_non_list() -> None:
    with pytest.raises(ValueError):
        _parse_tests('{"a": 1}')
