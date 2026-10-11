"""The system prompt and the first user message for one task.

The system prompt is the same for every task, so providers can cache it. Everything
task-specific goes in the first user message, built from an `AgentTask` only: the
agent never sees patches, tests or the eval script. Phase 3 adds context
strategies at `first_message`, so the rest of the loop stays the same.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent.harness.dataset import AgentTask
from agent.harness.sandbox import WORKDIR

SYSTEM_PROMPT = f"""\
You are a software engineer fixing a GitHub issue in a Python repository. The \
repository is checked out at {WORKDIR}, with its dependencies installed and its \
Python environment active. There is no network access.

Your job is to make the smallest change to the repository's source code that \
resolves the issue. Hidden tests will check your change after you finish. Do not \
modify existing tests; they are reset before grading.

Tools:
- read: show a file with line numbers, or list a directory.
- edit: replace one exact, unique piece of text in a file.
- write: create a file or overwrite it.
- bash: run a shell command in {WORKDIR}. Each call starts a fresh shell.
- submit: finish. Your changes to the files are your answer.

How to work:
1. Find the code involved: search with `grep -rn` and read the relevant files.
2. Reproduce the bug when you can, with a short script or a single test, so you \
can confirm the fix.
3. Make the fix. Keep it focused on the issue; do not refactor unrelated code.
4. Rerun your reproduction and the existing tests closest to the change. Run \
single test files or tests, not the whole suite, since commands time out.
5. Delete any scratch files you created, then call submit.

You have a limited number of turns, so act rather than describe what you plan to \
do, and call submit once the fix is in place."""


def first_message(task: AgentTask) -> str:
    """Build the first user message: the repository and the issue text."""
    return (
        f"Repository: {task.repo} (checked out at {WORKDIR})\n\n"
        f"<issue>\n{task.problem_statement.strip()}\n</issue>\n\n"
        "Resolve this issue."
    )


def initial_messages(task: AgentTask) -> list[BaseMessage]:
    """Return the messages that start a run: the system prompt and the issue."""
    return [SystemMessage(SYSTEM_PROMPT), HumanMessage(first_message(task))]
