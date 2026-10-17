"""The context strategies, and the registry `--context` picks from."""

from __future__ import annotations

import time

from agent.context.base import Context, ContextStrategy
from agent.context.repo import RepoView
from agent.harness.dataset import AgentTask


class NoContext:
    """The Phase 2 baseline: no context, so the first message is unchanged."""

    name = "none"

    def build(self, task: AgentTask, repo: RepoView, budget_chars: int) -> Context:
        """Return an empty context."""
        return Context(strategy=self.name, text="")


STRATEGIES: dict[str, ContextStrategy] = {s.name: s for s in (NoContext(),)}


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
