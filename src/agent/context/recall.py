"""Whether chosen files include the ones the gold patch changes.

Recall is computed after the fact, from `Task.gold_patch`: it scores a ranking or a
context, and never feeds into one. Nothing here may be called by a strategy.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.M)


def changed_files(diff: str) -> list[str]:
    """Return the paths a diff changes, in order (the new path of a rename)."""
    return list(dict.fromkeys(new for _, new in _DIFF_HEADER.findall(diff)))


def recall(chosen: Iterable[str], gold: Iterable[str]) -> float | None:
    """Return the share of `gold` paths in `chosen`; None if there are none."""
    wanted = set(gold)
    if not wanted:
        return None
    return len(wanted & set(chosen)) / len(wanted)
