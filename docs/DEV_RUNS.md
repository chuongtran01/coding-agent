# Dev Runs (Phase 2)

As of 2026-10-11. The agent built in Phase 2 ([PHASE2.md](PHASE2.md)), run on the five dev tasks ([DEV_TASKS.md](DEV_TASKS.md)) with `uv run harness solve --dev`, and every failed run read by hand (task 6). Runs are saved under `runs/` (git-ignored); run ids are given so they can be reloaded with `agent.runs.load_run`.

## Baseline result

This is the number Phase 3 compares against. Gemini 2.5 Flash (`google_genai:gemini-2.5-flash`), default limits (30 turns, 1M tokens, 30 min wall clock, 120 s per `bash` command), the code as of commit `da6ce08`: the prompt from `cb19f19`, the `read` cap at 16,000 characters, and `pytest-dev__pytest-7432` in place of `pallets__flask-4992`. Three batches: `dev-g25-v4-1`, `-2`, `-3`.

| Instance | Resolved | Turns | Mean cost | Mean runtime | Stops |
| --- | --- | --- | --- | --- | --- |
| `psf__requests-3362` | 1/3 | 24, 17, 16 | $0.057 | 123 s | submitted ×3 |
| `pytest-dev__pytest-7432` | 2/3 | 6, 8, 8 | $0.020 | 52 s | submitted ×3 |
| `pytest-dev__pytest-5227` | 3/3 | 4, 4, 12 | $0.006 | 24 s | submitted ×3 |
| `pytest-dev__pytest-7220` | 1/3 | 25, 30, 23 | $0.105 | 121 s | submitted, max_turns, max_tokens |
| `mwaskom__seaborn-3010` | 3/3 | 4, 9, 8 | $0.008 | 32 s | submitted ×3 |
| **All** | **10/15 (67%)** | | **$0.039/task** | **70 s/task** | |

The batches scored 4/5, 3/5 and 3/5, and no run was flagged by the isolation checks. Runtime is the whole `solve`: sandbox start, the agent loop, diff capture and evaluation. Cost is computed from each turn's token usage with the price table in `src/agent/model.py`, so cache reads are priced as such (Gemini caches implicitly; 59-68% of input tokens were cache reads).

**For the Phase 5 budget:** at the dev-task mean, the other 295 Lite tasks would cost about $11.50 and take about 5.7 hours run one at a time. Treat that as a floor: the dev tasks were picked for small repos and short test runs, and cost per task varies about 70-fold between them ($0.002 to $0.15), driven by how long the agent works and how much it reads.

### Before the task swap and the lower read cap

The same model and prompt with the old task set and a 40,000-character `read` cap (batches `dev-g25-v3-1` to `-3`) scored 9/15 (60%) at $0.044 and 79 s per task: `psf__requests-3362` 2/3, `pallets__flask-4992` 0/3, `pytest-dev__pytest-5227` 3/3, `pytest-dev__pytest-7220` 1/3, `mwaskom__seaborn-3010` 3/3.

On the four tasks in both sets the score went from 9/12 to 8/12, a single `psf__requests-3362` run, within the noise (that task has flipped before with nothing changed). The lower cap cut `pytest-dev__pytest-7220`'s mean tokens by about 7% (833k to 778k) and its mean cost by about 12%, but one of its runs still hit the token budget: its largest output was now an unfiltered `git grep -n "AssertionError"` over the repo (about 19,700 characters, just under the `bash` cap), followed by three capped reads. The cause is the history growing every turn, which lower caps only trim; managing what stays in context is Phase 3 and later work.

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

Read from the `v3` batches, plus the one `pytest-dev__pytest-7432` failure from `v4`.

- **`pytest-dev__pytest-7432` (1 failure in 3): a regression.** In `dev-g25-v4-2` the fix passes the hidden test but breaks the existing `test_xfail_run_anyway`: it moved the `--runxfail` "don't interfere" branch below the xfail handling, so with `--runxfail` that handling now interferes. The agent checked its fix with its own reproduction script and never ran `testing/test_skipping.py`, which the prompt asks for, so it never saw the break.

- **`pallets__flask-4992` (0/3, since replaced): not solvable from the issue.** The issue asks for a file mode parameter and suggests `mode="b"`; the agent adds `mode` every time. The hidden test calls `from_file(..., text=False)`, the name upstream chose. Expect this task to fail for any agent that follows the issue; consider swapping it for a task whose hidden test follows from its issue.
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

- The context grows every turn, and long runs (`pytest-dev__pytest-7220`) still hit the token budget after the `read` cap was lowered. Lower caps only trim the edges; what stays in context is a Phase 3+ question.
- One run per task is noisy (`psf__requests-3362` went resolved, unresolved, resolved with no change in between). Compare changes on at least three batches.
- A Claude comparison (`--model anthropic:claude-sonnet-5-5`), suggested in [PHASE2.md](PHASE2.md), needs an Anthropic key.
