# Dev Runs (Phase 2)

> **After these runs:** `pallets__flask-4992` was replaced by `pytest-dev__pytest-7432` (see [DEV_TASKS.md](DEV_TASKS.md)), and the `read` cap was lowered from 40,000 to 16,000 characters. The numbers below are for the old task set and cap; the baseline needs re-measuring before Phase 3 compares against it.

As of 2026-10-11. The agent built in Phase 2 ([PHASE2.md](PHASE2.md)), run on the five dev tasks ([DEV_TASKS.md](DEV_TASKS.md)) with `uv run harness solve --dev`, and every failed run read by hand (task 6). Runs are saved under `runs/` (git-ignored); run ids are given so they can be reloaded with `agent.runs.load_run`.

## Baseline result

Gemini 2.5 Flash (`google_genai:gemini-2.5-flash`), default limits (30 turns, 1M tokens, 30 min wall clock, 120 s per `bash` command), the prompt as of commit `cb19f19`. Three batches: `dev-g25-v3-1`, `-2`, `-3`.

| Instance | Resolved | Turns | Mean cost | Mean runtime | Stops |
| --- | --- | --- | --- | --- | --- |
| `psf__requests-3362` | 2/3 | 30, 14, 15 | $0.056 | 120 s | submitted ×3 |
| `pallets__flask-4992` | 0/3 | 18, 15, 30 | $0.037 | 87 s | submitted ×2, max_turns |
| `pytest-dev__pytest-5227` | 3/3 | 4, 4, 4 | $0.002 | 15 s | submitted ×3 |
| `pytest-dev__pytest-7220` | 1/3 | 29, 22, 30 | $0.119 | 147 s | submitted, max_tokens, max_turns |
| `mwaskom__seaborn-3010` | 3/3 | 5, 4, 11 | $0.006 | 26 s | submitted ×3 |
| **All** | **9/15 (60%)** | | **$0.044/task** | **79 s/task** | |

Each batch scored 3/5. Runtime is the whole `solve`: sandbox start, the agent loop, diff capture and evaluation. Cost is computed from each turn's token usage with the price table in `src/agent/model.py`, so cache reads are priced as such (Gemini caches implicitly; 57-72% of input tokens in these batches were cache reads).

**For the Phase 5 budget:** at the dev-task mean, the other 295 Lite tasks would cost about $13 and take about 6.5 hours run one at a time. Treat that as a floor: the dev tasks were picked for small repos and short test runs, and cost per task varies 50-fold between them ($0.002 to $0.18), driven by how long the agent works and how much it reads.

## How the prompt got here

The first batch (`dev-gemini-1`, the task 3 prompt) scored 2/5 at $0.014/task. Reading its failures and the runs that followed found four problems in the prompt, none in the tools:

1. **A bug phrased as a question was answered with documentation.** `psf__requests-3362` asks "Have I misunderstood something?"; the agent edited a docstring and submitted after 4 turns. Fix: the fix must change the code's behaviour; editing only docs or comments does not resolve an issue.
2. **Unverified submits.** Two of three failures submitted without running anything. Fix: rerun the reproduction to check the fix.
3. **`grep -rn` matched binary files in `.git` and `__pycache__`.** Fix: search with `git grep -n`.
4. **A deadlock, introduced by a first version of fix 2.** "Do not submit a change you have not run" combined with "rerun the closest existing tests" and "do not modify tests" left no way out when the issue changes behaviour that existing tests check: in `pytest-dev__pytest-5227` the agent made the correct one-line fix, five old-format tests failed, and it investigated until the turn limit. Fix: tests that check the old behaviour may fail, as expected; the hidden tests replace them.

With fixes 1-4, the same model went from 2/5 to 9/15, and every easy task stayed at 4-11 turns.

## Gemini 3.8 Flash

Tried as a stronger model (`dev-g38-1`, stopped after 3 runs; `dev-g38-v2-1`; single runs `g38-fix-1`, `g38-turns-2`). It scored 2/5 in its one full batch at $0.23/task, five times 2.5 Flash's cost, and used all 30 turns in 9 of its 10 runs:

- It investigates for most of the budget: its first edit came at turn 14-24, or never (5 of 10 runs ended with an empty diff). Several runs spent 10-20 turns in `git log -S` and `git show`, looking for the fix in history (the images hold no commits past `base_commit`; see [DEV_TASKS.md](DEV_TASKS.md)). The prompt now says the fix is not in the history.
- It never called `submit` once its fix was in, except in one run; the runs it resolved were scored because a limit still ends in a scored diff.
- It ignored a turn budget: each step's last tool result now ends with `[Turn N of 30 used; M left.]`, with a warning to make the change and submit once 5 or fewer are left. 3.8 Flash received the warning and its reasoning never mentioned it.

Since three signals (the prompt's turn limit, a target of about half the budget, the countdown) did not change its pacing, it is the model's behaviour, not a harness problem. 2.5 Flash is the development model again; 3.8 Flash stays selectable with `--model`. The turn note is kept: it is correct information for the agent and costs a few tokens.

## Failures under the current prompt

- **`pallets__flask-4992` (0/3): not solvable from the issue.** The issue asks for a file mode parameter and suggests `mode="b"`; the agent adds `mode` every time. The hidden test calls `from_file(..., text=False)`, the name upstream chose. Expect this task to fail for any agent that follows the issue; consider swapping it for a task whose hidden test follows from its issue.
- **`pytest-dev__pytest-7220` (1/3): hard, and the fixes are wrong.** The agent reproduces the bug every time, then tries fixes in `code.py`, `nodes.py` or `terminal.py` that its reproduction shows do not work, and runs out of turns or tokens. In `dev-g25-v3-2` it read the 1,217-line `src/_pytest/_code/code.py` whole three times; each read hit the 40,000-character cap (about 10k tokens), and since every turn resends the history, the 1M-token budget ran out at turn 22.
- **`psf__requests-3362` (1 failure in 3): a real but wrong fix.** Since fix 1 the agent changes behaviour in `iter_content` rather than docs, but one run iterated through 9 edits without finding the change in `stream_decode_response_unicode` and submitted at turn 30.

These are the model's reasoning, which Phase 2 does not tune.

## What the runs showed about the harness

- **Limits stop runs cleanly.** Seen in real runs: `max_turns` (many), `max_tokens` (`dev-g25-v3-2` `pytest-7220`: 1,025k tokens against a 1M budget, the expected overshoot of one turn, since limits are checked between turns), and `error` (`g38-turns-1`: the provider's 402 "credits depleted" ended the run at once, and its empty diff was still scored and saved). Each recorded its stop reason.
- **Tool errors are recoverable.** Across all runs the only tool errors were 8 `edit` failures: 7 "`old` was not found" and 1 "`old` and `new` are the same". The agent re-read the file and retried; no run stalled on them.
- **Truncation was hit, but did not explain a failure.** `bash` output was cut from the middle 9 times in failed runs, mostly long `git log -p` output from 3.8 Flash and test logs. The failures above are explained without the cut parts, though the cuts were not checked call by call.
- **Scratch files reach the diff.** Reproduction scripts (`test_path_error.py`, `noxfile.py`) appear in diffs when a run ends on a limit before the agent cleans up. Harmless to scoring.
- **Reasoning is now visible.** Gemini hides its reasoning unless asked; `make_model` requests thought summaries, which is how the 3.8 Flash deadlock above was diagnosed. They are verbose and open with filler ("Okay, here's my summary...").
- **No run saw the answer.** Every run now records two checks (`agent.isolation`, task 7): hidden text the model received before writing it itself, and paths to the task's package outside `/testbed`, such as the newer `requests` copies in `site-packages` (the known gap in [DEV_TASKS.md](DEV_TASKS.md)). Rerun over all 23 saved runs, neither flags anything. Getting there took three rounds of false positives, each a rule now in the check: lines the issue itself contains, lines the repo already has (the `requests` gold patch copies a comment block from elsewhere in `models.py`), and lines from old commits (`git log -p` showed the 2012 commit that first added `stream_decode_response_unicode`, with a line the fix reuses). A flagged run is marked `FLAGGED` by `solve` and its score should not be trusted.

## Open

- The `read` cap is now 16,000 characters (was 40,000): long files filled the context and the token budget fast. Its effect is not measured yet.
- One run per task is noisy (`psf__requests-3362` went resolved, unresolved, resolved with no change in between). Compare changes on at least three batches.
- A Claude comparison (`--model anthropic:claude-sonnet-5-5`), suggested in [PHASE2.md](PHASE2.md), needs an Anthropic key.
