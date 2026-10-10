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

## Runtime (gold patch)

Measured 2026-10-10 on an Apple Silicon Mac (Docker Desktop 29.8.2, 15 CPUs and 8 GB given to Docker), with the published x86_64 images running emulated. Each run: start the sandbox, apply the gold patch and `test_patch`, then run `pytest -rA` on the `.py` files `test_patch` touches. Range over two runs:

| Instance | Image size | Pull (once) | Container start | Test run | Total | Listed tests passed |
| --- | --- | --- | --- | --- | --- | --- |
| `psf__requests-3362` | 2.68 GB | 27 s | 1.3-1.8 s | 2.0-8.6 s | 4-13 s | 76 / 76 |
| `pallets__flask-4992` | 2.94 GB | 52 s | 0.7-1.5 s | 1.9-2.6 s | 4-6 s | 19 / 19 |
| `pytest-dev__pytest-5227` | 2.69 GB | 26 s | 0.7-1.1 s | 2.4-3.2 s | about 5 s | 34 / 34 (+3 junk ids, below) |
| `pytest-dev__pytest-7220` | 2.68 GB | 33 s | 0.7-0.9 s | 1.0-1.3 s | 3-4 s | 12 / 12 |
| `mwaskom__seaborn-3010` | about 2.7 GB | about 30 s | 0.9-1.5 s | 6.1-8.5 s | 8-12 s | 3 / 3 |

Emulation is fast enough at about 3-13 s per evaluation, so there is no need to build arm64 images. Image pulls dominate a first run; keep images between runs.

## Findings for Tasks 4-5

- **Run test files, not test ids.** Parametrized ids containing `/` or `::` (for example `test_ischildnode[foo/bar::TestBaz-foo/bar-False]`) cannot be selected on the pytest command line. Run the `.py` files from `test_patch` (skip non-Python files such as `tests/static/config.toml`) and read per-test status from the `-rA` output, as SWE-bench does.
- **Unlisted tests can fail.** `psf__requests-3362` reports 108 errors from tests that need a local httpbin server. None are in FAIL_TO_PASS or PASS_TO_PASS, so scoring must look only at the listed tests, never at pytest's exit code.
- **Junk ids in the dataset.** `pytest-dev__pytest-5227` PASS_TO_PASS contains `[`, `[100%]` and `[100%]------------------------------`, fragments of pytest's progress output. A naive "every listed test passed" check would fail the gold patch. Reuse SWE-bench's own log parser and grading so these match the way they did upstream.
- **Images add one commit.** `/testbed` has a `SWE-bench` commit on top of `base_commit` that only changes file modes. The sandbox accepts `base_commit` at `HEAD` or at `HEAD~1` under that commit. Diff an agent's work against `HEAD`, not `base_commit`, or the mode changes leak into the patch.
