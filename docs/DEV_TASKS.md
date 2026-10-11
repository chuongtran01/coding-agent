# Dev Tasks

As of 2026-10-09. The 3-5 SWE-bench Lite tasks used to build and validate the Phase 1 harness (see [PHASE1.md](PHASE1.md)). The list lives in code as `DEV_TASK_IDS` in `src/agent/harness/dataset.py`.

All five come from the `test` split, because SWE-bench publishes per-instance images for it. Agent work will be tuned on them, so **exclude them from Phase 5 benchmark numbers** (score on the other 295).

| Instance | FAIL_TO_PASS | PASS_TO_PASS | Why |
| --- | --- | --- | --- |
| `psf__requests-3362` | 1 | 75 | Pure Python, small dependency set; single target test |
| `pallets__flask-4992` | 1 | 18 | Pure Python, short test run |
| `pytest-dev__pytest-5227` | 3 | 34 | Pure Python; several FAIL_TO_PASS tests exercise per-test parsing |
| `pytest-dev__pytest-7220` | 1 | 11 | Smallest PASS_TO_PASS set among quick-install repos |
| `mwaskom__seaborn-3010` | 1 | 2 | Different repo shape (numpy/pandas/matplotlib stack), very short run |

Selection rule: repos without compiled extensions or heavy builds (excluding astropy, matplotlib, scikit-learn, xarray, django), then the fewest PASS_TO_PASS tests to keep runs short, and at least one task with several FAIL_TO_PASS tests.

Swap a task out if its image or tests fight you; record the replacement and reason here.

## Runtime and validation

Measured 2026-10-10 on an Apple Silicon Mac (Docker Desktop 29.8.2, 15 CPUs and 8 GB given to Docker), with the published x86_64 images running emulated, using `uv run harness validate`: two gold runs, one empty-patch run and one broken-patch run per task. All 20 runs scored as expected. A run covers container start, `git apply` of the candidate, the official eval script (reinstall, `test_patch`, tests), scoring and teardown.

| Instance | Image size | Pull (once) | Gold: tests | Gold: total | Empty: total | Broken: total | Listed tests (gold) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `psf__requests-3362` | 2.68 GB | 27 s | 2.8-2.9 s | 3.7-3.9 s | 3.3 s | 1.1 s | F2P 1/1, P2P 75/75 |
| `pallets__flask-4992` | 2.94 GB | 52 s | 3.1 s | 4.0-4.1 s | 3.7 s | 0.9 s | F2P 1/1, P2P 18/18 |
| `pytest-dev__pytest-5227` | 2.69 GB | 26 s | 4.7-5.0 s | 5.5-5.9 s | 5.3 s | 0.9 s | F2P 3/3, P2P 34/34 |
| `pytest-dev__pytest-7220` | 2.68 GB | 33 s | 4.2 s | 5.1-5.2 s | 4.5 s | 1.0 s | F2P 1/1, P2P 11/11 |
| `mwaskom__seaborn-3010` | about 2.7 GB | about 30 s | 4.2-4.3 s | 5.0-5.2 s | 4.7 s | 0.8 s | F2P 1/1, P2P 2/2 |

Emulation is fast enough at about 4-6 s per evaluation, so there is no need to build arm64 images. Image pulls dominate a first run; keep images between runs. The empty patch fails exactly the FAIL_TO_PASS tests and keeps every PASS_TO_PASS test passing; the broken patch (one removed line altered) stops at `git apply` and reports git's error.

## Findings for Tasks 4-5

- **Use the official eval script.** `SWE-bench/SWE-bench_Lite` (the SWE-bench team's copy, same 300 tasks as `princeton-nlp/SWE-bench_Lite`, with corrected test lists for `sympy__sympy-13177`) ships an `eval_script`, `image` and `log_parser` per task. The script resets the test files to `base_commit`, applies `test_patch`, reinstalls the package where needed (`requests` uses a non-editable `pip install .`), and runs the repo's own test command (Django uses `./tests/runtests.py`, not pytest). `runner.py` runs it unchanged and parses the log with SWE-bench's `get_logs_eval`. With it, a full evaluation takes 4-6 s per dev task, including the reinstall.
- **Run test files, not test ids.** Parametrized ids containing `/` or `::` (for example `test_ischildnode[foo/bar::TestBaz-foo/bar-False]`) cannot be selected on the pytest command line. The eval scripts already run whole files.
- **Unlisted tests can fail.** `psf__requests-3362` reports 108 errors from tests that need a local httpbin server. None are in FAIL_TO_PASS or PASS_TO_PASS, so scoring must look only at the listed tests, never at pytest's exit code.
- **Junk ids in the dataset.** `pytest-dev__pytest-5227` PASS_TO_PASS contains `[`, `[100%]` and `[100%]------------------------------`, fragments of pytest's progress output. SWE-bench's own parser produces the same fragments from the log, so they pass with the gold patch; a hand-written parser would not.
- **Images add one commit.** `/testbed` has a `SWE-bench` commit on top of `base_commit` that only changes file modes. The sandbox accepts `base_commit` at `HEAD` or at `HEAD~1` under that commit. Diff an agent's work against `HEAD`, not `base_commit`, or the mode changes leak into the patch.

## Isolation and limits (Task 7)

Checked 2026-10-10 on all five dev tasks; `tests/integration_tests/test_isolation.py` repeats the checks.

**What the agent's sandbox cannot reach**

- **The fix in git.** After applying the gold patch and `test_patch`, none of the resulting file versions exist as git objects in a fresh container. There are no remotes, no unreachable objects, and every ref (tags included) is an ancestor of `HEAD`, so the image holds no future commits.
- **The fix on disk.** The most distinctive lines the gold patch and `test_patch` add appear in no file in the container (whole filesystem searched), except for the `requests` gap below. Patch files and the eval script are written to `/tmp`, and only in the evaluation container.
- **The network.** `Sandbox.start` disables networking by default, so the agent cannot fetch the upstream fix or a newer release. `evaluate()` starts a separate, fresh container with networking on, because the eval scripts reinstall the package and fetch build tools (`setuptools`, `flit_core`) from PyPI. Offline, that reinstall fails; editable installs still pass, but `psf__requests-3362` (installed with `pip install .`) tests the old code and the gold patch scores not resolved.
- **Hidden fields on the host.** `Task.for_agent()` returns an `AgentTask` with only `instance_id`, `repo`, `base_commit` and `problem_statement`. Phase 2 hands the agent that, never the `Task`.

**Known gap: newer `requests` releases in the images.** pip vendors its own `requests` (2.31.0 and 2.32.3) and conda's base environment installs `requests` 2.31.0, all outside the testbed environment. They include upstream fixes, so for the 6 `psf/requests` tasks in Lite an agent could find the fix in, for example, `/opt/miniconda3/lib/python3.11/site-packages/requests/models.py`. No other dev repo has a copy of its package outside `/testbed`, and pip vendors no other Lite repo. The images are left unchanged so results stay comparable with the official leaderboard, which uses the same images. In Phase 2, flag runs where the agent reads files of the task's package outside `/testbed`.

**Limits on every run**

| Limit | Value | Where |
| --- | --- | --- |
| Memory (no swap on top) | 4 GB | `Sandbox.start`, `mem_limit` and `memswap_limit` |
| CPU | 2 cores | `Sandbox.start`, `nano_cpus` |
| Processes | 1024 | `Sandbox.start`, `pids_limit` |
| Network | off (on for evaluation only) | `Sandbox.start`, `network_mode` |
| Host mounts | none | `Sandbox.start` |
| Command timeout | 300 s default, 60 s for `git apply` and `mkdir`, 1800 s for the eval script | `Sandbox.exec` (coreutils `timeout`, SIGKILL 5 s later) |
| Container lifetime | 4 h; an orphaned container stops by itself | `Sandbox.start`, `sleep` as the main process |
| Cleanup | container removed on exit, even after an exception | `Sandbox.start`; `remove_leftover_containers()` after a crash |

Disk usage is not limited: Docker Desktop's storage driver does not support per-container size limits.
