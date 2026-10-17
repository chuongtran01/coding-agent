"""Render files with line numbers, whole or as the parts that match an issue.

`full_file` shows its files as the agent's `read` does. A file too big for what is
left of the budget is cut along Python's own boundaries, from `ast`: its imports,
then the top-level functions, methods and module-level names that best match the
issue's terms, each whole. A method comes with its class's `class` line (and
decorators), so the agent sees where it lives; not the class docstring, which in
some repos runs to hundreds of lines. A function is never cut in the middle; one
too big for the room left is left out.

A definition scores for being named by the issue (`NAME_MATCH` times the term's
weight) and for the terms it uses, with diminishing returns as in the ranking: the
i-th strongest counts 1/i, so a long function that touches many terms does not
outrank the one the issue names. A class's methods get a share of its name's
points, and its constructor (`__init__`, `__new__`) all of them, since an issue
that names a class usually calls it.

Line numbers count newline-separated lines, as `ast` does. They differ from `read`'s
only in files with form feeds or other line breaks Python does not count.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from agent.tools import MAX_LINE_CHARS

NAME_MATCH = 3.0  # a definition named after a term, against 1 for using it
MIN_SCORE = 0.5  # a full-weight term, or several prose words; one alone is no match
CLASS_SHARE = 0.5  # share of a class's name points each of its methods gets
CONSTRUCTORS = frozenset({"__init__", "__new__"})
GAP = "   ..."  # stands for lines left out of an excerpt

_WORD = re.compile(r"\b[A-Za-z_]\w*\b")
_IDENTIFIER = re.compile(r"^\w+$")
_QUOTES = re.compile(r"""[rRuUbB]*("{3}|'{3}|"|')""")


@dataclass(frozen=True)
class Definition:
    """A function, method, class header or named constant, scored against the issue."""

    lines: frozenset[int]  # all of it and, for a method, its class's header
    outline: frozenset[int]  # its signature and first docstring line, for skeletons
    score: float
    first: int


def whole(path: str, text: str) -> str:
    """Return the file with a header and line numbers, as `read` shows it."""
    lines = split_lines(text)
    return f"{path} ({len(lines)} lines)\n{numbered(lines, range(1, len(lines) + 1))}"


def excerpt(path: str, text: str, terms: dict[str, float], budget: int) -> str | None:
    """Return the file's imports and best-matching definitions in `budget` chars.

    `terms` are the issue's identifiers and their weights. Returns None when no
    definition matches, none fits, or the file does not parse.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):  # Python 2 syntax, or a null byte
        return None
    lines = split_lines(text)
    header = (
        f"{path} ({len(lines)} lines; showing its imports and the definitions "
        "that match the issue)\n"
    )
    imports = import_lines(tree)
    units = sorted(
        (u for u in definitions(tree, lines, terms) if u.score >= MIN_SCORE),
        key=lambda u: (-u.score, u.first),
    )
    room = budget - len(header)
    shown = imports if len(numbered(lines, imports)) <= room else set()
    chosen = False
    for unit in units:
        candidate = shown | unit.lines
        if len(numbered(lines, candidate)) <= room:
            shown, chosen = candidate, True
    if not chosen:
        return None
    return header + numbered(lines, shown)


def numbered(lines: list[str], shown: set[int] | range, *, gaps: bool = True) -> str:
    """Return the `shown` lines (1-based), numbered, with `GAP` where lines are left out.

    Without `gaps`, only the line numbers show what is left out, which keeps a
    skeleton, mostly gaps, half the size.
    """
    rows: list[str] = []
    previous = 0
    for n in sorted(shown):
        if gaps and rows and n > previous + 1:
            rows.append(GAP)
        line = lines[n - 1]
        if len(line) > MAX_LINE_CHARS:
            line = (
                line[:MAX_LINE_CHARS] + f"... [{len(line) - MAX_LINE_CHARS} chars cut]"
            )
        rows.append(f"{n:6}\t{line}")
        previous = n
    return "\n".join(rows)


def split_lines(text: str) -> list[str]:
    """Split a file into lines as `ast` numbers them."""
    lines = [line.removesuffix("\r") for line in text.split("\n")]
    return lines[:-1] if lines and lines[-1] == "" else lines


def import_lines(tree: ast.Module) -> set[int]:
    """Return the lines of the module's top-level imports."""
    return {
        n
        for node in tree.body
        if isinstance(node, ast.Import | ast.ImportFrom)
        for n in _span(node)
    }


def definitions(
    tree: ast.Module, lines: list[str], terms: dict[str, float]
) -> list[Definition]:
    """Return the module's functions, classes, methods and term-named constants.

    Only top-level definitions and the methods of top-level classes: nested
    functions are part of the function they are in.
    """
    units = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            units.append(_unit(node, node.name, lines, terms))
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in terms:
                    units.append(_unit(node, target.id, lines, terms))
        elif isinstance(node, ast.ClassDef):
            head = _header(node)
            named = NAME_MATCH * terms.get(node.name, 0.0)
            units.append(
                Definition(
                    lines=head,
                    outline=head | _docstring_line(node, lines),
                    score=named,
                    first=min(head),
                )
            )
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    method = _unit(item, item.name, lines, terms)
                    share = 1.0 if item.name in CONSTRUCTORS else CLASS_SHARE
                    units.append(
                        Definition(
                            lines=method.lines | head,
                            outline=method.outline | head,
                            score=method.score + share * named,
                            first=method.first,
                        )
                    )
    return units


def _unit(
    node: ast.stmt, name: str, lines: list[str], terms: dict[str, float]
) -> Definition:
    """Score a definition: its name matching a term, and the terms it uses."""
    span = _span(node)
    source = "\n".join(lines[n - 1] for n in sorted(span))
    words = set(_WORD.findall(source))
    used = sorted(
        (
            weight
            for term, weight in terms.items()
            if (term in words if _IDENTIFIER.match(term) else term in source)
        ),
        reverse=True,
    )
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        outline = _header(node) | _docstring_line(node, lines)
    else:
        outline = frozenset({node.lineno})
    return Definition(
        lines=span,
        outline=outline,
        score=NAME_MATCH * terms.get(name, 0.0)
        + sum(weight / i for i, weight in enumerate(used, 1)),
        first=min(span),
    )


def _span(node: ast.stmt) -> frozenset[int]:
    """Return a statement's lines, its decorators included."""
    decorators = getattr(node, "decorator_list", [])
    first = min([node.lineno, *(d.lineno for d in decorators)])
    return frozenset(range(first, (node.end_lineno or node.lineno) + 1))


def _header(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> frozenset[int]:
    """Return a definition's decorators and `class` or `def` lines, up to its body."""
    first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
    # A body that starts on the `def` line leaves the header that one line.
    end = max(node.body[0].lineno - 1, node.lineno)
    return frozenset(range(first, end + 1))


def _docstring_line(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef, lines: list[str]
) -> frozenset[int]:
    """Return the line holding the first words of a docstring, if there is one."""
    doc = node.body[0]
    if not (
        isinstance(doc, ast.Expr)
        and isinstance(doc.value, ast.Constant)
        and isinstance(doc.value.value, str)
    ):
        return frozenset()
    for n in range(doc.lineno, (doc.end_lineno or doc.lineno) + 1):
        if _QUOTES.sub("", lines[n - 1]).strip():
            return frozenset({n})
    return frozenset()
