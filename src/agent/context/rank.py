"""Rank the repo's Python files by how strongly the issue points at them.

The ranking is shared by every strategy that shows code, so strategies differ only
in how they render the files it returns. It is rules, not a model: terms are
pulled from the issue text, then each tracked `.py` file is scored on

    path in issue      the issue names the file outside a traceback (a bare name
                       like `utils.py` is split between the files it could be)
    traceback          the file is a frame in a traceback; the last frame inside
                       the repo, nearest the error, scores again. Weaker than a
                       path someone wrote out: long tracebacks name every layer
    module             a dotted name is its module (`requests.utils`)
    name               a word of the issue, or two adjacent words joined, is the
                       file's name ("blueprint" -> `blueprints.py`, "index
                       entries" -> `indexentries.py`)
    defines            it defines a term: `def`, `class`, or a module-level
                       assignment; split between the files that define it
    mentions           it uses a term; rarer terms count for more, and each term
                       adds at most `MENTION` however often it appears

Identifiers only the issue's code blocks contain (logs, tracebacks, the user's
script) count half, except in an exception's message, which names the culprit
(`RST.__init__() got an unexpected keyword argument`). Plain words of the prose count as weak terms, so an issue
written without any code still ranks files; hyphenated names such as
`If-Modified-Since` are searched as written and in snake case, and command-line
options (`--fixtures`) as written. Test files score half: existing tests near the bug can help,
but the fix is in source. The same issue and repo always give the same ranking.
"""

from __future__ import annotations

import builtins
import keyword
import math
import re
from collections import defaultdict
from collections.abc import Iterable

from pydantic import BaseModel

from agent.context.repo import RepoView
from agent.harness.dataset import AgentTask

PATH = 10.0
TRACEBACK = 6.0
LAST_FRAME = 4.0
MODULE = 6.0
NAME = 4.0
DEFINES = 5.0
MENTION = 1.0
CODE_ONLY = 0.5  # weight of identifiers found only inside code blocks
PROSE = 0.3  # weight of plain words of the prose
TEST_FILE = 0.5  # factor on a test file's score
MAX_MENTION_REASONS = 5

_STOP = (
    set(keyword.kwlist)
    | set(dir(builtins))
    | {"self", "cls", "args", "kwargs", "py", "test", "tests", "foo", "bar", "baz"}
)
_CODE_BLOCK = re.compile(r"```.*?(?:```|\Z)", re.S)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_PATH_TOKEN = re.compile(r"[\w.~/\\-]*\w\.py\b")
_FRAMES = (
    re.compile(r'File "([^"\n]+\.py)", line \d+'),  # CPython traceback
    re.compile(r"File (\S+\.py):\d+"),  # IPython traceback
    re.compile(r"(?<![\w\"])([\w.~/\\-]+\.py):\d+"),  # pytest, compilers: path:line
)
_DOTTED = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
_IDENTIFIER = re.compile(r"^\w+$")
_HYPHENATED = re.compile(
    r"\b[A-Z][A-Za-z0-9]*(?:-[A-Z][A-Za-z0-9]*)+\b"
)  # Content-Type
_OPTION = re.compile(r"(?<![\w-])--[A-Za-z][\w-]+")
_EXCEPTION_MESSAGE = re.compile(
    r"^\s*[\w.]*(?:Error|Exception|Warning):[ \t]*(.+)$", re.M
)
_PROSE_WORD = re.compile(r"[A-Za-z]{4,}")
# Common English words, which say nothing about where the code is.
_ENGLISH = set(
    """
    about above after again also always another anything around because been
    before being below between both cannot case could does doing done down during
    each either else even every example expected expect fail fails false following
    from further getting given good have having here however instead into issue
    itself just know like line little look made make many maybe might more most
    much must need never no none note only other otherwise output over people
    please possible problem rather really result right same seems should since
    some something still such sure take than that their them then there these
    they thing think this those though through thus time true under until upon
    used uses using very want wants well were what when where whether which while
    will with within without work works would wrong yeah your
    actual actually behavior behaviour bug code current currently error errors
    file files fine function happen happens instead raise raises return returns
    run running seem show shows thanks version want way when
    """.split()
)
_DUNDER = re.compile(r"^__\w+__$")  # `__init__` and the like are defined everywhere
_WORD = re.compile(r"\b[A-Za-z_]\w*\b")
_CALL = re.compile(r"\b([A-Za-z_]\w*)\(")
_DEFINITION = re.compile(
    r"^\s*(?:async\s+)?(?:def|class)\s+(\w+)|^(\w+)\s*(?::[^=]*)?="
)


class IssueTerms(BaseModel):
    """What the issue names: file paths, traceback frames, modules, identifiers."""

    paths: list[str]  # tokens that end in .py, outside traceback frames
    frames: list[str]  # traceback frames, in order, innermost last
    modules: list[str]  # dotted names, which may be module paths
    identifiers: dict[str, float]  # name -> weight (1, CODE_ONLY or PROSE)
    words: list[str] = []  # the prose's plain words, lowercased, in order


class RankedFile(BaseModel):
    """A file, its score, and the evidence behind the score."""

    path: str
    score: float
    reasons: list[str]


def issue_terms(text: str) -> IssueTerms:
    """Pull file paths, traceback frames, dotted names and identifiers from an issue."""
    frames = [m.group(1) for line in text.splitlines() for m in _first_frame(line)]
    outside_frames = text
    for pattern in _FRAMES:
        outside_frames = pattern.sub(" ", outside_frames)
    paths = _unique(_PATH_TOKEN.findall(outside_frames))
    modules = _unique(d for d in _DOTTED.findall(text) if not d.endswith(".py"))

    # Paths are matched as paths: their parts (`src`, `_pytest`) are not identifiers.
    text = _PATH_TOKEN.sub(" ", text)
    code = "\n".join(_CODE_BLOCK.findall(text))
    prose = _CODE_BLOCK.sub("\n", text)
    named = "\n".join(_INLINE_CODE.findall(prose))
    prose = _INLINE_CODE.sub(" ", prose)

    identifiers: dict[str, float] = {}

    def add(name: str, weight: float) -> None:
        identifiers[name] = max(identifiers.get(name, 0.0), weight)

    words = [w.lower() for w in _PROSE_WORD.findall(prose) if w.lower() not in _ENGLISH]
    for word in words:
        if word not in _STOP:
            add(word, PROSE)
    for option in _OPTION.findall(text):
        add(option, 1.0)
    for message in _EXCEPTION_MESSAGE.findall(text):
        for name in _identifiers(message, every_word=False):
            add(name, 1.0)
    for hyphenated in _HYPHENATED.findall(text):
        add(hyphenated, 1.0)
        add(hyphenated.replace("-", "_").lower(), CODE_ONLY)
    for source, weight in ((code, CODE_ONLY), (prose, 1.0), (named, 1.0)):
        for name in _identifiers(source, every_word=source is named):
            add(name, weight)
    return IssueTerms(
        paths=paths,
        frames=frames,
        modules=modules,
        identifiers=identifiers,
        words=words,
    )


def rank_files(task: AgentTask, repo: RepoView) -> list[RankedFile]:
    """Return the repo's Python files that the issue points at, best first."""
    files = repo.python_files()
    terms = issue_terms(task.problem_statement)
    scores: dict[str, list[tuple[float, str]]] = defaultdict(list)

    for token in terms.paths:
        matches = resolve_path(token, files)
        for path in matches:
            scores[path].append((PATH / len(matches), "path in issue"))
    in_repo = [m for frame in terms.frames if (m := resolve_path(frame, files))]
    for path in {p for matches in in_repo for p in matches}:
        scores[path].append((TRACEBACK, "in traceback"))
    if in_repo and len(in_repo[-1]) == 1:
        scores[in_repo[-1][0]].append((LAST_FRAME, "last repo frame in traceback"))
    for dotted in terms.modules:
        for path in _module_files(dotted, files):
            scores[path].append((MODULE, "module named in issue"))

    for path, word in _name_matches(terms.words, files):
        scores[path].append((NAME, f"name matches '{word}'"))

    identifiers = terms.identifiers
    if identifiers:
        _score_definitions(repo, identifiers, scores)
        _score_mentions(repo, identifiers, len(files), scores)

    ranked = []
    for path, parts in scores.items():
        parts = _diminish(_best_per_reason(parts))
        score = sum(points for points, _ in parts)
        if is_test_file(path):
            score *= TEST_FILE
        reasons = [reason for _, reason in sorted(parts, key=lambda p: (-p[0], p[1]))]
        ranked.append(
            RankedFile(path=path, score=round(score, 4), reasons=_trim(reasons))
        )
    return sorted(ranked, key=lambda f: (-f.score, f.path))


def resolve_path(token: str, files: list[str]) -> list[str]:
    """Return the tracked files a path from the issue can refer to.

    An exact path wins; then a longer path that ends in a tracked one (an absolute
    path, or one into site-packages); then tracked files that end in the token (a
    bare `skipping.py`).
    """
    token = token.replace("\\", "/").removeprefix("./")
    if token in files:
        return [token]
    longer = [f for f in files if token.endswith("/" + f)]
    if longer:
        return [max(longer, key=len)]
    return [f for f in files if f.endswith("/" + token)]


def is_test_file(path: str) -> bool:
    """Whether `path` looks like a test module or test helper."""
    *dirs, name = path.split("/")
    return (
        bool({"test", "tests", "testing"} & set(dirs))
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def _score_definitions(
    repo: RepoView,
    identifiers: dict[str, float],
    scores: dict[str, list[tuple[float, str]]],
) -> None:
    names = "|".join(sorted(n for n in identifiers if _IDENTIFIER.match(n)))
    if not names:
        return
    matches = repo.grep(
        [
            rf"^[[:space:]]*(async[[:space:]]+)?(def|class)[[:space:]]+({names})[[:space:]]*[(:]",
            rf"^({names})[[:space:]]*(:[^=]*)?=([^=]|$)",
        ],
        regex=True,
    )
    defined_in: dict[str, set[str]] = defaultdict(set)
    for match in matches:
        found = _DEFINITION.match(match.text)
        name = found and (found.group(1) or found.group(2))
        if name in identifiers:
            defined_in[name].add(match.path)
    for name, paths in defined_in.items():
        points = DEFINES * identifiers[name] / math.sqrt(len(paths))
        for path in paths:
            scores[path].append((points, f"defines {name}"))


def _score_mentions(
    repo: RepoView,
    identifiers: dict[str, float],
    n_files: int,
    scores: dict[str, list[tuple[float, str]]],
) -> None:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for match in repo.grep(sorted(identifiers), word=True, only_matching=True):
        if match.text in identifiers:
            counts[(match.path, match.text)] += 1
    files_with: dict[str, int] = defaultdict(int)
    for _, name in counts:
        files_with[name] += 1
    for (path, name), count in counts.items():
        # Rarer terms say more: 1 for a term in one file, 0 for one in every file.
        rarity = (
            math.log(n_files / files_with[name]) / math.log(n_files)
            if n_files > 1
            else 1.0
        )
        points = MENTION * identifiers[name] * rarity * min(count, 3) / 3
        if points > 0:
            scores[path].append((points, f"mentions {name} ×{count}"))


def _name_matches(words: list[str], files: list[str]) -> list[tuple[str, str]]:
    """Return (file, words) where a word or adjacent pair of words is the file's name.

    A file's name is its module name (a package's for `__init__.py`), compared
    without underscores and a trailing "s". A name that fits more than a few files
    (`utils`, `base`) is too common to count.
    """
    by_name: dict[str, list[str]] = defaultdict(list)
    for path in files:
        parts = path.removesuffix(".py").split("/")
        name = parts[-2] if parts[-1] == "__init__" and len(parts) > 1 else parts[-1]
        by_name[_name_key(name)].append(path)
    phrases = words + [a + b for a, b in zip(words, words[1:], strict=False)]
    found: dict[str, str] = {}
    for phrase in phrases:
        paths = by_name.get(_name_key(phrase), [])
        if 0 < len(paths) <= 3:
            for path in paths:
                found.setdefault(path, phrase)
    return sorted(found.items())


def _name_key(name: str) -> str:
    key = name.replace("_", "").lower()
    return key[:-1] if key.endswith("s") and len(key) > 4 else key


def _module_files(dotted: str, files: list[str]) -> list[str]:
    """Return files whose module path is a prefix (of 2+ parts) of `dotted`."""
    parts = dotted.split(".")
    found = []
    for n in range(len(parts), 1, -1):
        base = "/".join(parts[:n])
        for suffix in (f"{base}.py", f"{base}/__init__.py"):
            found += [f for f in files if f == suffix or f.endswith("/" + suffix)]
        if found:
            return found
    return []


def _identifiers(text: str, *, every_word: bool) -> list[str]:
    """Return the names in `text` that look like code.

    That is snake_case and CamelCase words, calls, and the parts of dotted names
    (the first only when capitalised, as a class is); in inline code
    (`every_word`), every word counts. Dunder names are defined everywhere, so
    they are dropped.
    """
    names = set(_CALL.findall(text))
    for dotted in _DOTTED.findall(text):
        first, *rest = dotted.split(".")
        # The first part is usually a variable (`r`, `np`), unless it is a class.
        names |= set(rest) | ({first} if first[0].isupper() else set())
    for word in _WORD.findall(text):
        if every_word or "_" in word.strip("_") or re.search(r"[a-z][A-Z]", word):
            names.add(word)
    return sorted(
        n
        for n in names
        if len(n) >= 3 and n not in _STOP and not n.isdigit() and not _DUNDER.match(n)
    )


def _first_frame(line: str) -> list[re.Match[str]]:
    for pattern in _FRAMES:
        matches = list(pattern.finditer(line))
        if matches:
            return matches
    return []


def _best_per_reason(parts: list[tuple[float, str]]) -> list[tuple[float, str]]:
    best: dict[str, float] = {}
    for points, reason in parts:
        best[reason] = max(best.get(reason, 0.0), points)
    return [(points, reason) for reason, points in best.items()]


def _diminish(parts: list[tuple[float, str]]) -> list[tuple[float, str]]:
    """Give many weak terms diminishing returns: the i-th strongest of a kind counts 1/i.

    Otherwise a file that defines ten names from a traceback's code lines outranks
    the file the traceback ends in.
    """
    out: list[tuple[float, str]] = []
    seen: dict[str, int] = defaultdict(int)
    for points, reason in sorted(parts, key=lambda p: (-p[0], p[1])):
        kind = reason.split(" ")[0]
        if kind in ("defines", "mentions"):
            seen[kind] += 1
            points /= seen[kind]
        out.append((points, reason))
    return out


def _trim(reasons: list[str]) -> list[str]:
    mentions = [r for r in reasons if r.startswith("mentions ")]
    dropped = len(mentions) - MAX_MENTION_REASONS
    kept = [r for r in reasons if r not in mentions[MAX_MENTION_REASONS:]]
    return kept + [f"and {dropped} more terms"] * (dropped > 0)


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))
