"""A file's skeleton for `repo_map`: imports, signatures and first docstring lines.

A skeleton shows where things are without their bodies: the module's top-level
imports, every top-level function, class and method signature (decorators
included) with its real line numbers, and the line holding the first words of
each docstring. It is a small fraction of the file, so `repo_map` covers several
times as many files as `full_file` in the same budget, and the agent reads the
bodies it needs with `read`.

A skeleton too big for the room it is given keeps its imports and the definitions
that best match the issue, scored as `excerpt` scores them. A file `ast` cannot
parse (Python 2 syntax) falls back to its `import`, `def` and `class` lines.
"""

from __future__ import annotations

import ast
import re
from typing import NamedTuple

from agent.context.excerpt import definitions, import_lines, numbered, split_lines

_OUTLINE_LINE = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+\w|^(?:import|from)\s+\S")


class Skeleton(NamedTuple):
    """A rendered skeleton, and how it was made."""

    text: str
    trimmed: bool  # only the definitions that best match the issue
    fallback: bool  # from `def`, `class` and `import` lines; the file does not parse


def skeleton(
    path: str, text: str, terms: dict[str, float], budget: int | None = None
) -> Skeleton | None:
    """Return the skeleton of a file, cut to `budget` characters if one is given.

    `terms` are the issue's identifiers and their weights, which choose what a cut
    skeleton keeps. Returns None for a file with no definitions or imports, or when
    nothing fits.
    """
    lines = split_lines(text)
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):  # Python 2 syntax, or a null byte
        return _fallback(path, lines, budget)
    imports = import_lines(tree)
    found = definitions(tree, lines, terms)
    every = imports.union(*(d.outline for d in found))
    if not every:
        return None
    full = (
        f"{path} ({len(lines)} lines; skeleton: imports, signatures and first "
        f"docstring lines)\n{numbered(lines, every, gaps=False)}"
    )
    if budget is None or len(full) <= budget:
        return Skeleton(full, trimmed=False, fallback=False)

    header = (
        f"{path} ({len(lines)} lines; skeleton of its imports and the definitions "
        "that best match the issue)\n"
    )
    room = budget - len(header)
    shown = imports if len(numbered(lines, imports, gaps=False)) <= room else set()
    chosen = False
    for definition in sorted(found, key=lambda d: (-d.score, d.first)):
        candidate = shown | definition.outline
        if len(numbered(lines, candidate, gaps=False)) <= room:
            shown, chosen = candidate, True
    if not chosen:
        return None
    return Skeleton(
        header + numbered(lines, shown, gaps=False), trimmed=True, fallback=False
    )


def _fallback(path: str, lines: list[str], budget: int | None) -> Skeleton | None:
    """Return a file's `def`, `class` and `import` lines, in order, within `budget`."""
    header = (
        f"{path} ({len(lines)} lines; skeleton from its def, class and import "
        "lines: it does not parse)\n"
    )
    outline = [n for n, line in enumerate(lines, 1) if _OUTLINE_LINE.match(line)]
    room = None if budget is None else budget - len(header)
    shown: set[int] = set()
    for n in outline:
        if room is not None and len(numbered(lines, shown | {n}, gaps=False)) > room:
            break
        shown.add(n)
    if not shown:
        return None
    return Skeleton(
        header + numbered(lines, shown, gaps=False),
        trimmed=len(shown) < len(outline),
        fallback=True,
    )
