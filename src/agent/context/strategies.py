"""The context strategies, and the registry `--context` picks from."""

from __future__ import annotations

import time

from agent.context.base import Context, ContextItem, ContextStrategy
from agent.context.excerpt import excerpt, whole
from agent.context.rank import issue_terms, rank_files
from agent.context.repo import RepoError, RepoView
from agent.harness.dataset import AgentTask


class NoContext:
    """The Phase 2 baseline: no context, so the first message is unchanged."""

    name = "none"

    def build(self, task: AgentTask, repo: RepoView, budget_chars: int) -> Context:
        """Return an empty context."""
        return Context(strategy=self.name, text="")


MAX_FILES = 10  # past the top ten, the ranking is mostly weak mentions
MIN_ROOM = 500  # chars; less than this left of the budget shows nothing useful
TRIMMED_SHARE = 0.5  # of the room left, the most a trimmed file may take
SEPARATOR = "\n\n"


class FullFile:
    """The top-ranked files, whole and numbered, until the budget is used.

    A file too big for what is left is shown as its imports and the definitions
    that best match the issue (`excerpt`), and marked `trimmed`. It gets at most
    half the room left: a large top file would otherwise fill the budget with
    weak matches and push out the next files, where the fix often is (the
    ranking's top file is right about two times in three, its top three four in
    five). A file with nothing that fits is passed over.
    """

    name = "full_file"

    def build(self, task: AgentTask, repo: RepoView, budget_chars: int) -> Context:
        """Return the top-ranked files, whole or trimmed, within `budget_chars`."""
        terms = issue_terms(task.problem_statement).identifiers
        blocks: list[str] = []
        items: list[ContextItem] = []
        used = 0
        for ranked in rank_files(task, repo)[:MAX_FILES]:
            room = budget_chars - used - len(SEPARATOR) * bool(blocks)
            if room < MIN_ROOM:
                break
            try:
                text = repo.read(ranked.path)
            except RepoError:
                continue
            if not text.strip() or "\0" in text:
                continue
            full = whole(ranked.path, text)
            trimmed = len(full) > room
            block = (
                excerpt(ranked.path, text, terms, int(room * TRIMMED_SHARE))
                if trimmed
                else full
            )
            if block is None:
                continue
            blocks.append(block)
            used = len(SEPARATOR.join(blocks))
            items.append(
                ContextItem(
                    path=ranked.path,
                    kind="file",
                    trimmed=trimmed,
                    score=ranked.score,
                    reasons=ranked.reasons,
                    chars=len(block),
                )
            )
        return Context(strategy=self.name, text=SEPARATOR.join(blocks), items=items)


STRATEGIES: dict[str, ContextStrategy] = {s.name: s for s in (NoContext(), FullFile())}


class UnknownStrategyError(ValueError):
    """A context strategy name that is not in `STRATEGIES`."""

    def __init__(self, name: str) -> None:
        """Name the unknown strategy and list the known ones."""
        super().__init__(
            f"unknown context strategy {name!r}; choose from {', '.join(STRATEGIES)}"
        )


def get_strategy(name: str) -> ContextStrategy:
    """Return the strategy called `name`, or raise `UnknownStrategyError`."""
    try:
        return STRATEGIES[name]
    except KeyError:
        raise UnknownStrategyError(name) from None


def build_context(
    strategy: ContextStrategy, task: AgentTask, repo: RepoView, budget_chars: int
) -> Context:
    """Run `strategy` and record how long it took."""
    start = time.monotonic()
    context = strategy.build(task, repo, budget_chars)
    return context.model_copy(update={"build_s": time.monotonic() - start})
