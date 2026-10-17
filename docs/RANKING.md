# Ranking Recall (Phase 3, task 3)

As of 2026-10-16. The shared file ranking (`src/agent/context/rank.py`) measured before any strategy or agent run, as [PHASE3.md](PHASE3.md) task 3 requires. Reproduce with `uv run harness rank --dev` and `uv run harness rank --tuning`; no API calls.

**Ranking recall@k** is the share of files the gold patch changes that are among the top k ranked files. It is computed from `Task.gold_patch` after ranking; the ranking never sees it. Every gold patch here changes one file, so recall@k is the share of tasks whose gold file is in the top k.

## Result

| Set | Tasks | @1 | @3 | @5 | @10 |
| --- | --- | --- | --- | --- | --- |
| Tuning | 25 | 0.64 | 0.80 | 0.84 | 0.92 |
| Dev | 5 | 0.60 | 0.60 | 0.60 | 0.80 |
| **All** | **30** | **0.63** | **0.77** | **0.80** | **0.90** |

The tuning tasks are 25 Lite tasks drawn by repo with a fixed seed (`TUNING_TASK_IDS`), because the dev tasks have no django or sympy, which are 191 of Lite's 300 tasks. Rules were changed after seeing them, so they are excluded from Phase 5 like the dev tasks ([DEV_TASKS.md](DEV_TASKS.md)). Ranking takes 2-40 s per task, nearly all of it starting the container.

**For the strategies:** with recall@3 near 0.8 and @10 near 0.9, `full_file` (two or three files in a 24k-character budget) gets the gold file most of the time, and `repo_map` (a dozen or more) covers a few more. Neither can help the 10% below.

## Where the gold file ranks

| Task | Rank | Why |
| --- | --- | --- |
| `astropy__astropy-14182` | 2 | Five traceback frames outrank it; the exception message names its class `RST` |
| `django__django-13590` | 1 | |
| `django__django-14382` | 1 | |
| `django__django-14672` | 1 | |
| `django__django-15347` | 1 | |
| `django__django-15498` | 7 | No code in the issue; found through the header name `If-Modified-Since` |
| `django__django-15814` | 2 | |
| `matplotlib__matplotlib-23964` | 1 | |
| `matplotlib__matplotlib-25332` | 46 | The fix is in `cbook.Grouper`, which the issue never names; it is reached only through `align_labels` |
| `mwaskom__seaborn-3190` | 1 | |
| `pallets__flask-4045` | 1 | No code in the issue; "blueprint" matches `blueprints.py` |
| `psf__requests-1963` | 1 | |
| `pydata__xarray-5131` | 1 | |
| `pylint-dev__pylint-7228` | 67 | The traceback names the config loading path; the fix is in the regex option converter it calls |
| `pytest-dev__pytest-5221` | 7 | The issue names the option `--fixtures`, mentioned in many files |
| `pytest-dev__pytest-7168` | 2 | |
| `scikit-learn__scikit-learn-10508` | 1 | |
| `scikit-learn__scikit-learn-13439` | 1 | |
| `sphinx-doc__sphinx-7975` | 2 | No code in the issue; "index entries" matches `indexentries.py` |
| `sphinx-doc__sphinx-8595` | 1 | |
| `sympy__sympy-11400` | 1 | |
| `sympy__sympy-12481` | 1 | |
| `sympy__sympy-18532` | 1 | |
| `sympy__sympy-21171` | 1 | |
| `sympy__sympy-24066` | 4 | |
| `psf__requests-3362` (dev) | 7 | The fix is in `stream_decode_response_unicode`, which `iter_content` calls; the issue names only `iter_content` |
| `pytest-dev__pytest-7432` (dev) | 1 | |
| `pytest-dev__pytest-5227` (dev) | 1 | |
| `pytest-dev__pytest-7220` (dev) | 33 | Nothing in the issue points at `nodes.py` |
| `mwaskom__seaborn-3010` (dev) | 1 | |

**The misses are mostly one kind:** the fix is in a function the issue never names, one call away from one it does (`requests-3362`, `matplotlib-25332`, `pylint-7228`, and probably `pytest-7220`). Rules over the issue's words cannot reach those; following calls from the named functions can, which is Phase 4's graph traversal. These four tasks are a natural first test for it.

## How the rules got here

Each round was run on both sets; rules stayed generic (no repo-specific rules).

| Round | Change | Tuning @1 / @3 / @5 / @10 | Dev @1 / @3 / @5 / @10 |
| --- | --- | --- | --- |
| 1 | Paths, traceback frames, dotted modules, definitions and mentions of identifiers, rarity-weighted; test files halved | | 0.40 / 0.60 / 0.60 / 0.80 |
| 2 | Path parts (`src`, `_pytest`) are not identifiers; many weak terms get diminishing returns (the i-th strongest counts 1/i), so ten names from a traceback's code lines no longer outrank the frame the error is in | 0.60 / 0.68 / 0.76 / 0.80 | 0.60 / 0.60 / 0.60 / 0.80 |
| 3 | A capitalised head of a dotted name is a class (`RST.__init__`); dunder names dropped; plain prose words as weak terms, hyphenated names (`If-Modified-Since`) as literals, file names matched against words and word pairs. Three prose-only issues had ranked nothing | 0.64 / 0.72 / 0.80 / 0.92 | unchanged |
| 4 | A path seen only in a traceback scores as a frame, not also as a path named in the issue; names in an exception's message count fully; command-line options (`--fixtures`) as literals | 0.64 / 0.80 / 0.84 / 0.92 | unchanged |

Stopped there: the remaining misses need call-following, and more rules would start to fit these 30 tasks.
