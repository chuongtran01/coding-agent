"""The context strategies, and the registry `--context` picks from."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal, NamedTuple

from agent.context.base import Context, ContextItem, ContextStrategy
from agent.context.excerpt import excerpt, whole
from agent.context.rank import issue_terms, rank_files
from agent.context.repo import RepoError, RepoView
from agent.context.skeleton import skeleton
from agent.harness.dataset import AgentTask


class NoContext:
    """The Phase 2 baseline: no context, so the first message is unchanged."""

    name = "none"

    def build(self, task: AgentTask, repo: RepoView, budget_chars: int) -> Context:
        """Return an empty context."""
        return Context(strategy=self.name, text="")


MAX_FILES = 10  # full_file: past the top ten, the ranking is mostly weak mentions
MAX_MAP_FILES = 30  # repo_map: skeletons are small, so it looks further down
MIN_ROOM = 500  # chars; less than this left of the budget shows nothing useful
TRIMMED_SHARE = 0.5  # of the room left, the most a trimmed file may take
SEPARATOR = "\n\n"


class _Rendered(NamedTuple):
    text: str
    trimmed: bool
    fallback: bool = False


# Renders one ranked file in the room given (path, text, issue terms, room), or
# returns None to pass over it.
_Render = Callable[[str, str, dict[str, float], int], _Rendered | None]


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
        return _fill(self.name, "file", _full_file, MAX_FILES, task, repo, budget_chars)


class RepoMap:
    """Skeletons of the top-ranked files (`skeleton`), until the budget is used.

    A skeleton is a small fraction of its file, so the map covers more files than
    `full_file`, without their bodies. Breadth is the point, so every skeleton,
    not only one too big for what is left, gets at most half the room: the
    skeleton of a large module runs to 17k characters (`sympy/printing/latex.py`)
    and would leave room for little else. One over that keeps the definitions
    that best match the issue and is marked `trimmed`.
    """

    name = "repo_map"

    def build(self, task: AgentTask, repo: RepoView, budget_chars: int) -> Context:
        """Return skeletons of the top-ranked files within `budget_chars`."""
        return _fill(
            self.name, "skeleton", _repo_map, MAX_MAP_FILES, task, repo, budget_chars
        )


def _full_file(
    path: str, text: str, terms: dict[str, float], room: int
) -> _Rendered | None:
    full = whole(path, text)
    if len(full) <= room:
        return _Rendered(full, trimmed=False)
    cut = excerpt(path, text, terms, int(room * TRIMMED_SHARE))
    return None if cut is None else _Rendered(cut, trimmed=True)


def _repo_map(
    path: str, text: str, terms: dict[str, float], room: int
) -> _Rendered | None:
    outline = skeleton(path, text, terms)
    if outline is not None and len(outline.text) > room * TRIMMED_SHARE:
        outline = skeleton(path, text, terms, int(room * TRIMMED_SHARE))
    return None if outline is None else _Rendered(*outline)


def _fill(
    strategy: str,
    kind: Literal["file", "skeleton"],
    render: _Render,
    max_files: int,
    task: AgentTask,
    repo: RepoView,
    budget_chars: int,
) -> Context:
    """Render the top-ranked files in order until the budget is used."""
    terms = issue_terms(task.problem_statement).identifiers
    blocks: list[str] = []
    items: list[ContextItem] = []
    used = 0
    for ranked in rank_files(task, repo)[:max_files]:
        room = budget_chars - used - len(SEPARATOR) * bool(blocks)
        if room < MIN_ROOM:
            break
        try:
            text = repo.read(ranked.path)
        except RepoError:
            continue
        if not text.strip() or "\0" in text:
            continue
        rendered = render(ranked.path, text, terms, room)
        if rendered is None:
            continue
        blocks.append(rendered.text)
        used = len(SEPARATOR.join(blocks))
        items.append(
            ContextItem(
                path=ranked.path,
                kind=kind,
                trimmed=rendered.trimmed,
                fallback=rendered.fallback,
                score=ranked.score,
                reasons=ranked.reasons,
                chars=len(rendered.text),
            )
        )
    return Context(strategy=strategy, text=SEPARATOR.join(blocks), items=items)


STRATEGIES: dict[str, ContextStrategy] = {
    s.name: s for s in (NoContext(), FullFile(), RepoMap())
}


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
