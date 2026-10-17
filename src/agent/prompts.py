"""The system prompt and the first user message for one task.

The system prompt is the same for every task, so providers can cache it. Everything
task-specific goes in the first user message, built from an `AgentTask` only: the
agent never sees patches, tests or the eval script. A context strategy's code goes
in the first message too, so the system prompt is the same for every strategy.
"""

from __future__ import annotations

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from agent.context import Context
from agent.harness.dataset import AgentTask
from agent.harness.sandbox import WORKDIR

SYSTEM_PROMPT = f"""\
You are a software engineer fixing a GitHub issue in a Python repository. The \
repository is checked out at {WORKDIR}, with its dependencies installed and its \
Python environment active. There is no network access, so packages cannot be \
installed.

Your job is to make the smallest change to the repository's source code that \
resolves the issue. Hidden tests will check the code's behaviour after you finish, \
so the fix must change how the code behaves: editing only docs or comments does not \
resolve an issue, even one phrased as a question. Do not modify existing tests; \
they are reset before grading.

Tools:
- read: show a file with line numbers, or list a directory.
- edit: replace one exact, unique piece of text in a file.
- write: create a file or overwrite it.
- bash: run a shell command in {WORKDIR}. Each call starts a fresh shell.
- submit: finish. Your changes to the files are your answer.

How to work:
1. Find the code involved: search with `git grep -n` (it skips `.git` and \
ignored files) and read the relevant files. The repository's history stops before \
the fix, so do not search `git log` for it.
2. Reproduce the bug when you can, with a short script or a single test, so you \
can confirm the fix.
3. Make the fix. Keep it focused on the issue; do not refactor unrelated code.
4. Rerun your reproduction to check the fix, then the existing tests closest to \
the change. Run single test files or tests, not the whole suite, since commands \
time out. If existing tests fail because they check the old behaviour that the \
issue asks you to change, that is expected: the hidden tests replace them. Do not \
edit them or undo your fix to make them pass.
5. Delete any scratch files you created, then call submit.

You have a limited number of turns, and each tool result ends with how many are \
left. Aim to have your change in place by about half of them, and use the rest to \
check it. Act rather than describe what you plan to do, and call submit once the \
fix is in place and checked."""


CONTEXT_NOTE = (
    "The code below was selected automatically from the issue. It may be "
    "incomplete, or include code unrelated to the fix."
)


def first_message(task: AgentTask, context: Context | None = None) -> str:
    """Build the first user message: the repository, the issue and any context.

    An empty context adds nothing, so the message is the Phase 2 one.
    """
    section = ""
    if context is not None and context.text:
        section = f"{CONTEXT_NOTE}\n\n<context>\n{context.text.strip()}\n</context>\n\n"
    return (
        f"Repository: {task.repo} (checked out at {WORKDIR})\n\n"
        f"<issue>\n{task.problem_statement.strip()}\n</issue>\n\n"
        f"{section}Resolve this issue."
    )


def initial_messages(
    task: AgentTask, context: Context | None = None
) -> list[AnyMessage]:
    """Return the messages that start a run: the system prompt, the issue, any context."""
    return [SystemMessage(SYSTEM_PROMPT), HumanMessage(first_message(task, context))]
