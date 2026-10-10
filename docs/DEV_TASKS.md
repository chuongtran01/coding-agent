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

Swap a task out if its image or tests fight you in Task 2 (for example, requests tests that need network access); record the replacement and reason here.
