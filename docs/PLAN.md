# Coding Agent: Build Plan

A single-pass, autonomous bug-fixing agent (Python, LangGraph). It takes a SWE-bench-lite issue, works in a sandbox, and iterates against real test output. Context strategy is a swappable plugin, and the result is a benchmark comparing strategies.

**Scope:** bug-fixing only, Python repos only, autonomous (no chat mode) until the benchmark is done.

**Timeline:** 6-10 weeks at full evenings/weekends, 10-14 weeks at 5-8 hrs/week.

**Apply-ready milestone:** after Phase 3 plus a partial Phase 5 (working agent, 2+ strategies, about 10 tasks benchmarked), start applying and keep building.

## Status

| Phase | Name | Status |
|---|---|---|
| 0 | Research | Not started |
| 1 | Sandbox + test runner (no AI) | Done |
| 2 | Bare agent loop | Done |
| 3 | Context strategy plugins | **In progress** |
| 4 | Graph traversal strategy | Not started |
| 5 | Eval harness + experiments | Not started |
| 6 | Polish | Not started |

Phase 0 reading can run alongside Phase 1 (read SWE-agent's eval harness first, since Phase 1 depends on it).

---

## Phase 1: Sandbox + test runner, no AI (3-5 days)

**Goal:** run one SWE-bench-lite task end-to-end programmatically, with a known-good patch, and get a correct pass/fail. No LLM involved yet.

**Done.** What was built, and where it differs from the plan below, is in [PHASE1.md](PHASE1.md); runtimes and findings are in [DEV_TASKS.md](DEV_TASKS.md).

### Tasks

1. **Load the dataset**
   - Pull SWE-bench-lite via Hugging Face `datasets` (`SWE-bench/SWE-bench_Lite`, the SWE-bench team's copy of `princeton-nlp/SWE-bench_Lite` with each task's image and eval script)
   - Write a `Task` model: `instance_id`, `repo`, `base_commit`, `problem_statement`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `gold_patch`, `test_patch`
   - Pick 3-5 small tasks to develop against, favoring repos that are quick to install

2. **Build the sandbox**
   - Docker container per attempt, fresh working copy checked out at `base_commit`
   - Install the repo's dependencies inside the container
   - Agent-visible filesystem is the container only, nothing on the host
   - Decide early: use SWE-bench's published per-instance images, or build your own (published images are faster to start and keep results comparable)

3. **Patch application**
   - `apply_patch(sandbox, diff)` using `git apply`
   - Return a structured result: applied or not, plus error text

4. **Test runner**
   - Apply the `test_patch` (adds the hidden FAIL_TO_PASS tests) only at evaluation time, never before the agent runs
   - Run FAIL_TO_PASS and PASS_TO_PASS tests, parse results per test
   - Return `TestResult`: per-test status, raw output (truncated), runtime

5. **Scoring**
   - Resolved = all FAIL_TO_PASS pass and all PASS_TO_PASS still pass
   - Reuse the official SWE-bench evaluation logic where possible so numbers stay comparable

6. **Validate the harness**
   - Run each dev task with the **gold patch**: must score resolved
   - Run each with an **empty patch**: must score not resolved
   - Run with a deliberately **broken patch**: must score not resolved, with sensible error output
   - If gold patch ever fails, fix the harness before moving on

7. **Isolation checks**
   - Confirm the hidden tests and gold patch are not readable from anything the agent will later see
   - Add timeouts and resource limits to every container run

### Definition of done
- One command runs a task + patch through the sandbox and prints resolved / not resolved
- Gold patch resolves 3-5 dev tasks reliably; empty and broken patches do not
- A run takes a reasonable, known amount of time per task (record it, it feeds the cost estimates later)

### Risks
- Dependency installs for old repo versions can be painful. Start with the easiest repos and skip any that fight you.
- Docker image size and disk usage add up quickly. Prune between runs.

---

## Phase 0: Research (3-5 days)

- Pi: the four core tools (read, write, edit, bash) and how `pi-agenticoding` does spawn / ledger / handoff
- Aider: repo-map implementation (parsing + graph ranking)
- SWE-agent: tool interface and eval harness (needed for Phase 1)
- OpenHands: skim for a full-featured contrast

For each, extract: exact tool signatures, how tool results are formatted back to the model, stopping conditions, and how the eval is run.

**Done when:** you can explain each project's tool design and eval setup without looking it up.

## Phase 2: Bare agent loop (1 week)

- Four tools: `read`, `write`, `edit`, `bash`, with a small system prompt
- LangGraph graph: explore, propose patch, apply and test, loop on failure (cap about 5 iterations)
- Agent sees only the issue text, never the FAIL_TO_PASS tests
- This is the **floor baseline**: no engineered context

**Done when:** the agent attempts a task end-to-end, even at a low pass rate.

## Phase 3: Context strategy plugins (1 week)

Details, tasks and open questions in [PHASE3.md](PHASE3.md).

- Define a `ContextStrategy` protocol so the agent loop is identical across conditions
- Implement `none`, `full_file`, and `repo_map` (keyword/import-ranked skeleton)
- Structured logging of every decision and the context selected at each step

**Done when:** swapping strategies is one config change.

## Phase 4: Graph traversal strategy (1-1.5 weeks)

- Build the code graph with Python's `ast` (tree-sitter only if needed): functions, classes, imports, calls, inheritance
- Store in `networkx`, traverse N hops from the entry point to select context
- Main scope-creep risk: keep to Python, accept imperfect call resolution

**Done when:** the strategy runs on real tasks and returns sensible context.

## Phase 5: Eval harness + experiments (1 week)

- 15-20 tasks across all conditions: none, full-file, repo-map, graph traversal, drawn from the 270 Lite tasks that are neither dev tasks nor ranking tuning tasks ([DEV_TASKS.md](DEV_TASKS.md))
- Record pass rate, iterations to pass, tokens, latency, cost
- Review failures by hand: why a strategy failed is the best interview material
- One extra condition: history handling (below), on the best starting strategy only

**Done when:** you have a comparison table you can explain.

### History handling (an extra Phase 5 condition)

Phases 3-4 choose what the agent starts with; this chooses what stays in the history as the run goes on. Phase 2 showed the need: every turn resends everything read so far, so a run's cost grows roughly with the square of its length, and `pytest-dev__pytest-7220` hit the 1M-token budget in two of six runs even after the `read` cap was lowered ([DEV_RUNS.md](DEV_RUNS.md)). It is mainly a cost and robustness lever: those runs also had wrong fixes, so its effect on the pass rate is unknown and has to be measured.

- A `HistoryStrategy` that transforms the messages sent to the model before each call; the saved trajectory stays complete
- `append`: today's behaviour, keep everything (the baseline)
- `mask`: replace tool outputs older than the last N turns with a short placeholder (`[output of read src/x.py, 412 lines, removed]`), keeping every tool call and its result so providers still pair them; no extra model calls, so build it first
- `summarize`: past a size threshold, have the model summarise older turns and replace them; costs an extra call each time
- Compare against `append` on tokens, cost, runs hitting `max_tokens`, cache-read share, and pass rate

It is kept out of Phases 3-4 because it changes the loop, which those comparisons hold fixed, and because it breaks the loop's append-only rule: rewriting history invalidates the provider's prompt cache (57-72% of input tokens were cache reads in Phase 2) and Claude's thinking blocks, so savings must be measured net of lost cache hits. Testing it on the best starting strategy only, rather than every combination, keeps the number of runs down.

**Done when:** `mask` (and `summarize` if time allows) is compared with `append` on the Phase 5 tasks, with the cost and pass-rate difference explained.

## Phase 6: Polish (3-5 days)

- README with architecture diagram, results table, honest limitations
- Tracing view you can screen-share (LangSmith / Studio or your own logs)
- Short write-up on what worked, what didn't, and the tradeoffs (indexing cost and staleness vs. precision)

**Done when:** someone can understand the project in 5 minutes.

---

## Out of scope for now
- Interactive / chat mode (possible v2)
- Multi-language support
- Feature implementation
