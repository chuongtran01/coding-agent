# Context Runs (Phase 3)

As of 2026-10-17. The three context strategies of [PHASE3.md](PHASE3.md), run on the five dev tasks ([DEV_TASKS.md](DEV_TASKS.md)) with Gemini 2.5 Flash, the Phase 2 prompt and limits, and the 24,000-character context budget. Three interleaved rounds (task 7): in each round every task ran under `none`, `full_file` and `repo_map` in turn, and the strategy going first rotated between rounds. Run with `uv run harness solve --dev --context none,full_file,repo_map --repeat 3 --run-id cmp`; batches `cmp-<round>-<strategy>`, 45 runs, $1.99 in all. No run crashed or was flagged by the isolation checks.

## Result

| Strategy | Resolved | Turns, mean (median) | Tokens, mean | Cost/task | Runtime/task | Context recall |
| --- | --- | --- | --- | --- | --- | --- |
| `none` | **14/15** | 13.7 (13) | 227k | $0.038 | 75 s | |
| `full_file` | 12/15 | 13.9 (11) | 350k | $0.057 | 93 s | 0.60 |
| `repo_map` | 10/15 | 12.8 (10) | 268k | $0.038 | 73 s | 0.80 |

| Task | `none` | `full_file` | `repo_map` |
| --- | --- | --- | --- |
| `psf__requests-3362` | 3/3 (24, 17, 14 turns) | 3/3 (30, 11, 10) | 2/3 (7, 10, 10) |
| `pytest-dev__pytest-7432` | 3/3 (7, 17, 10) | 3/3 (8, 9, 11) | 2/3 (8, 7, 10) |
| `pytest-dev__pytest-5227` | 3/3 (4, 3, 4) | 3/3 (2, 3, 8) | 3/3 (3, 3, 3) |
| `pytest-dev__pytest-7220` | 2/3 (30, 28, 12) | 0/3 (30, 19, 30) | 0/3 (29, 30, 28) |
| `mwaskom__seaborn-3010` | 3/3 (4, 13, 19) | 3/3 (13, 8, 17) | 3/3 (10, 4, 30) |

Context recall here is per run, so it differs from the 30-task measure in [RANKING.md](RANKING.md). The dev set has two tasks whose gold file is out of reach: `requests-3362`'s is ranked 7th, past what `full_file` shows, and `pytest-7220`'s 33rd.

**Read this as a check that the strategies work, not as a ranking of them.** `none` sends exactly the Phase 2 message, yet it scored 14/15 here against 10/15 in the Phase 2 baseline ([DEV_RUNS.md](DEV_RUNS.md)), with no change to the agent. That four-run swing is larger than any gap between strategies in this table. What the table does show:

- **Both strategies work end to end.** Every context stayed under the budget, and no context tripped the isolation checks.
- **On the four tasks other than `pytest-7220`:** `full_file` resolved 12/12, the same as `none`, at 10.8 turns against 11.3. `repo_map` resolved 10/12, in the fewest turns (8.8) and at the lowest cost ($0.021 a task, against $0.027 for `none`). Its two failures are model reasoning errors already seen under `none` in Phase 2 (below).
- **`pytest-7220`:** 0 of 6 runs with context resolved, against 2 of 3 without. Neither strategy shows its gold file. The one review that found the context to blame is below.
- **Cost:** a context adds about 8k input tokens to every request (the first request is 9.3k tokens against 1.6k for `none`). `full_file` cost 51% more per task than `none`; `repo_map` cost the same, because its runs were shorter. Gemini's implicit caching makes the repeated context cheap, but not free.

## Runs worse than `none`, read by hand

**`repo_map` on `psf__requests-3362` (round 1): a wrong, unverified fix.** The map listed `requests/utils.py`, and the agent was reading `stream_decode_response_unicode`, the function the gold patch changes, by its third turn. It then decided that reading `self.encoding` would set the encoding, added the bare expression `self.encoding` to `iter_content`, and checked it only with an existing test that already passed. It submitted at turn 7. The context got it to the right code faster than any `none` run; the reasoning that followed was wrong. This is the "real but wrong fix" failure in [DEV_RUNS.md](DEV_RUNS.md).

**`repo_map` on `pytest-dev__pytest-7432` (round 2): the known regression.** The agent removed the `--runxfail` "don't interfere" branch and broke `test_xfail_run_anyway`. It checked the fix with its own script and never ran `testing/test_skipping.py`. This is the same failure Phase 2 recorded under `none` (`dev-g25-v4-2`); the context did not cause it.

**`full_file` and `repo_map` on `pytest-dev__pytest-7220` (all six runs): the context was partly to blame.** The fix is in `src/_pytest/nodes.py`, which ranks 33rd: the issue is about a path shown wrongly after a fixture changes directory, and its words point at fixtures. Both strategies put `fixtures.py`, `pytester.py` and `monkeypatch.py` first.

- `repo_map` round 1 followed `fixtures.py` into `getfslineno` and `FixtureLookupError`, a different path-reporting route. It spent its remaining turns on edits to `src/_pytest/_code/code.py` it had to undo, and hit the token budget.
- The other five runs went where the `none` runs go: `code.py` is read 5-12 times per run, under every strategy. Two `full_file` runs did edit `nodes.py`, alongside wrong changes to `code.py`.
- `full_file` round 2 made no edit at all. It used up the 1M-token budget in 19 turns at 55k input tokens a turn. The context accounts for about 8k of that 55k; the rest is the files it read and kept rereading.

So on this task the context added a wrong lead and some token overhead to a task that was already hard: `none` resolved it once in three in the Phase 2 baseline.

## What this means for Phase 5

- **The comparison needs the larger set.** On five tasks, one run decides most of the gap between strategies, and `none` alone moved by four runs between two batches. Phase 5's 270 tasks are what can separate the strategies.
- **Report context recall next to each result.** On `pytest-7220` the context missed the gold file and the runs did worse. Splitting results by whether the context held the gold file separates a ranking problem from a strategy problem, as PHASE3.md's risk table plans.
- **Wrong leads are the risk to watch.** The context note says the code may be unrelated. On `pytest-7220` that warning did not stop one run from following the wrong lead.
