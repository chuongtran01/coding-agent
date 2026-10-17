"""The types every context strategy shares."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel

from agent.context.repo import RepoView
from agent.harness.dataset import AgentTask


class ContextItem(BaseModel):
    """One file a strategy put in the context, and why it chose it."""

    path: str
    kind: Literal["file", "skeleton"]
    trimmed: bool  # full_file: only the matching functions of a file too big to show
    score: float
    reasons: list[str]  # e.g. "defines iter_content", "path in traceback"
    chars: int


class Context(BaseModel):
    """What a strategy chose: the text for the first message, and its parts."""

    strategy: str
    text: str  # the section added to the first message; "" adds nothing
    items: list[ContextItem] = []
    build_s: float = 0.0  # set by `build_context`, not by the strategy


class ContextStrategy(Protocol):
    """Chooses code for the first message from the issue and the repo."""

    @property
    def name(self) -> str:
        """The name `AgentConfig.context` and `--context` select it by."""
        ...

    def build(self, task: AgentTask, repo: RepoView, budget_chars: int) -> Context:
        """Return the context for `task`, with `text` at most `budget_chars` long."""
        ...
