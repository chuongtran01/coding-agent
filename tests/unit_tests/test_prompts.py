from langchain_core.messages import HumanMessage, SystemMessage

from agent.context import Context
from agent.harness.dataset import Task
from agent.prompts import (
    CONTEXT_NOTE,
    SYSTEM_PROMPT,
    first_message,
    initial_messages,
)

TASK = Task(
    instance_id="psf__requests-0000",
    repo="psf/requests",
    base_commit="a" * 40,
    problem_statement="\n  `Session.get` drops the `timeout` argument.  \n",
    FAIL_TO_PASS=("test_x.py::test_secret_new_test",),
    PASS_TO_PASS=("test_x.py::test_a",),
    gold_patch="diff --git a/x.py b/x.py\n+GOLD_FIX_LINE\n",
    test_patch="diff --git a/test_x.py b/test_x.py\n+TEST_PATCH_LINE\n",
    version="2.0",
    image="swebench/sweb.eval.x86_64.psf_1776_requests-0000:latest",
    eval_type="pass_and_fail",
    log_parser="parse_log_requests",
    eval_script="#!/bin/bash\nEVAL_SCRIPT_LINE\n",
)


def test_first_message_has_repo_and_issue() -> None:
    msg = first_message(TASK.for_agent())
    assert msg.startswith("Repository: psf/requests (checked out at /testbed)\n")
    assert "<issue>\n`Session.get` drops the `timeout` argument.\n</issue>" in msg


PHASE2_FIRST_MESSAGE = (
    "Repository: psf/requests (checked out at /testbed)\n\n"
    "<issue>\n`Session.get` drops the `timeout` argument.\n</issue>\n\n"
    "Resolve this issue."
)


def test_no_context_is_exactly_the_phase2_message() -> None:
    # The `none` strategy reproduces the Phase 2 baseline byte for byte.
    agent_task = TASK.for_agent()
    assert first_message(agent_task) == PHASE2_FIRST_MESSAGE
    empty = Context(strategy="none", text="")
    assert first_message(agent_task, empty) == PHASE2_FIRST_MESSAGE


def test_context_goes_between_the_issue_and_the_request() -> None:
    context = Context(strategy="x", text="\nrequests/models.py\n1  import x\n")
    msg = first_message(TASK.for_agent(), context)
    assert msg == (
        "Repository: psf/requests (checked out at /testbed)\n\n"
        "<issue>\n`Session.get` drops the `timeout` argument.\n</issue>\n\n"
        f"{CONTEXT_NOTE}\n\n"
        "<context>\nrequests/models.py\n1  import x\n</context>\n\n"
        "Resolve this issue."
    )
    _, user = initial_messages(TASK.for_agent(), context)
    assert user.content == msg


def test_system_prompt_is_the_same_for_every_task() -> None:
    # Nothing task-specific, so the prefix caches across turns and tasks.
    for text in (TASK.repo, TASK.instance_id, TASK.base_commit):
        assert text not in SYSTEM_PROMPT


def test_system_prompt_names_every_tool() -> None:
    from agent.tools import TOOL_NAMES

    for name in TOOL_NAMES:
        assert f"- {name}:" in SYSTEM_PROMPT


def test_initial_messages() -> None:
    system, user = initial_messages(TASK.for_agent())
    assert isinstance(system, SystemMessage) and system.content == SYSTEM_PROMPT
    assert isinstance(user, HumanMessage)
    assert user.content == first_message(TASK.for_agent())


def test_prompts_contain_no_hidden_task_data() -> None:
    sent = "\n".join(str(m.content) for m in initial_messages(TASK.for_agent()))
    for hidden in (
        "GOLD_FIX_LINE",
        "TEST_PATCH_LINE",
        "EVAL_SCRIPT_LINE",
        "test_secret_new_test",
        TASK.base_commit,
    ):
        assert hidden not in sent
