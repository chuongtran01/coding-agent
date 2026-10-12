"""Check a run for the two ways the answer could reach the agent.

`hidden_text_seen` looks for text only the evaluation data has (the gold patch,
the test patch, the eval script) in what the model received. It should always be
empty: the agent is built from `AgentTask`, and the sandbox holds none of it. The
lines to look for come from `hidden_lines`, which needs the sandbox at its base
commit to drop lines the repo already contains.

`package_paths_outside_testbed` flags reads of the task's own package outside the
repo. The images ship newer releases of some packages (pip vendors `requests`; see
docs/DEV_TASKS.md), which can contain the upstream fix. Runs with such reads are
reported separately rather than trusted.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage

from agent.graph import ToolCall
from agent.harness.dataset import Task
from agent.harness.sandbox import WORKDIR
from agent.tools import Shell

MIN_HIDDEN_LINE = 25  # shorter added lines (imports, `return x`) occur all over a repo
TESTBED_ENV = "/opt/miniconda3/envs/testbed/"  # the task's environment: base code only

_EVAL_MARKERS = (r"EOF_\d{6,}", r">>>>> (?:Start|End) Test Output")
_CANDIDATES = "/tmp/.agent-hidden-lines"
# An absolute path: a slash not preceded by a path or URL character, so neither the
# tail of a relative path (`src/flask/x.py`) nor a URL (`https://...`) matches.
_PATH = re.compile(r"(?<![\w.\-+/:~])/[\w.\-+][\w.\-+/]*")


def candidate_lines(task: Task) -> list[str]:
    """Return text that the evaluation data adds, before checking it against the repo.

    These are the long lines the gold and test patches add, minus any the patches
    also show as existing (context or removed) lines or that the issue itself
    contains, plus the markers the eval script wraps around the test patch.
    """
    lines: set[str] = set()
    for diff in (task.gold_patch, task.test_patch):
        added, existing = set(), set()
        for line in diff.splitlines():
            if line.startswith(("+++", "---")):
                continue
            if line.startswith("+"):
                added.add(line[1:].strip())
            elif line.startswith((" ", "-")):
                existing.add(line[1:].strip())
        lines |= {line for line in added - existing if len(line) >= MIN_HIDDEN_LINE}
    lines = {line for line in lines if line not in task.problem_statement}
    for marker in _EVAL_MARKERS:
        lines |= set(re.findall(marker, task.eval_script))
    return sorted(lines)


def hidden_lines(task: Task, shell: Shell) -> list[str]:
    """Return the candidate lines that the repo and its history do not contain.

    A line the base code already has (the gold patch often copies a comment or an
    import) is not a leak when the agent reads it, and neither is one from an old
    commit (`git log -p` shows them); the images hold no commits past the base, so
    history cannot contain the fix. Searching every commit's patch covers the
    current files too, since each of their lines was added by some commit.
    """
    candidates = candidate_lines(task)
    if not candidates:
        return []
    shell.write_file(_CANDIDATES, "\n".join(candidates) + "\n")
    # -a: the log can contain binary bytes, and grep would then stop printing matches.
    result = shell.exec(
        f"git log --all -p --format= | grep -a -o -F -f {_CANDIDATES}; "
        f"rm -f {_CANDIDATES}",
        timeout=300,
    )
    in_repo = {line.strip() for line in result.output.splitlines()}
    return [line for line in candidates if line not in in_repo]


def hidden_text_seen(
    hidden: Sequence[str], messages: Sequence[BaseMessage]
) -> list[str]:
    """Return hidden lines the model received before it wrote them itself.

    An agent that writes the right fix sees its own lines echoed back (by `edit`,
    `git diff`, a test run); that is not a leak. A line counts only if a message
    the model received contains it before any of the model's replies does.
    """
    produced = ""
    seen: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage):
            produced += str(message.content)
            for call in message.tool_calls:
                produced += "\n".join(str(value) for value in call["args"].values())
            continue
        text = str(message.content)
        seen |= {line for line in hidden if line in text and line not in produced}
    return sorted(seen)


def package_names(task: Task) -> set[str]:
    """Return the import names of the task's package, e.g. {'pytest', '_pytest'}."""
    name = task.repo.split("/")[-1].replace("-", "_")
    return {name, f"_{name}"}


def package_paths_outside_testbed(
    task: Task, tool_calls: Sequence[ToolCall]
) -> list[str]:
    """Return paths into the task's package, outside the repo, that the agent touched.

    Both arguments and outputs count: a `grep -r /` can surface a file the agent
    never named. The testbed environment holds the base code, so it is allowed,
    except for copies vendored by pip, which are newer releases.
    """
    names = package_names(task)
    found: set[str] = set()
    for call in tool_calls:
        for path in _PATH.findall(json.dumps(call.args) + "\n" + call.output):
            parts = path.split("/")
            if not names & set(parts) or path.startswith(f"{WORKDIR}/"):
                continue
            if "_vendor" in parts or not path.startswith(TESTBED_ENV):
                found.add(path)
    return sorted(found)
