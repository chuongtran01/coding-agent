import json

import pytest
from pydantic import ValidationError

from agent.harness.dataset import Task

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
    "image": "swebench/sweb.eval.x86_64.psf_1776_requests-0000:latest",
    "eval_type": "pass_and_fail",
    "log_parser": "parse_log_requests",
    "eval_script": "#!/bin/bash\npytest -rA test_x.py\n",
}


def test_model_validate_maps_fields() -> None:
    task = Task.model_validate(ROW)
    assert task == Task(
        instance_id="psf__requests-0000",
        repo="psf/requests",
        base_commit="a" * 40,
        problem_statement="Something is broken.",
        FAIL_TO_PASS=("test_x.py::test_new",),
        PASS_TO_PASS=("test_x.py::test_a", "test_x.py::test_b"),
        gold_patch="diff --git a/x.py b/x.py\n",
        test_patch="diff --git a/test_x.py b/test_x.py\n",
        version="2.0",
        image="swebench/sweb.eval.x86_64.psf_1776_requests-0000:latest",
        eval_type="pass_and_fail",
        log_parser="parse_log_requests",
        eval_script="#!/bin/bash\npytest -rA test_x.py\n",
    )


def test_task_is_hashable_and_frozen() -> None:
    task = Task.model_validate(ROW)
    hash(task)
    with pytest.raises(ValidationError):
        task.repo = "other"


def test_test_lists_accept_list_and_empty() -> None:
    task = Task.model_validate(
        {**ROW, "FAIL_TO_PASS": ["a", "b"], "PASS_TO_PASS": "[]"}
    )
    assert task.FAIL_TO_PASS == ("a", "b")
    assert task.PASS_TO_PASS == ()


def test_test_lists_reject_non_list() -> None:
    with pytest.raises(ValidationError):
        Task.model_validate({**ROW, "FAIL_TO_PASS": '{"a": 1}'})


def test_base_commit_must_be_full_sha() -> None:
    with pytest.raises(ValidationError):
        Task.model_validate({**ROW, "base_commit": "abc123"})


def test_json_round_trip() -> None:
    task = Task.model_validate(ROW)
    assert Task.model_validate_json(task.model_dump_json()) == task
