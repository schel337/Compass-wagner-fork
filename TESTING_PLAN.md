# Test Plan for Compass

## 1. Codebase overview (what we're testing)

| Component | Location | Testability |
|---|---|---|
| CLI / orchestration | `compass/main.py` (1835 lines), `parseArgs()`, `entry()`, `compass_work()` | Medium (heavy side effects, subprocesses, multiprocessing) |
| Core algorithm | `compass/compass/algorithm.py` (`singleSampleCompass`, `maximize_reaction`), `penalties.py` (`eval_reaction_penalties`) | Medium — needs a model + expression + LP solves |
| Solver layer | `compass/opt/base.py` (`Optimizer`, `LinearProgramDelta`, `Solution`), `gurobi.py`, `cuopt.py` | **High** — clean abstract interface, ideal for fakes |
| Models / import | `compass/models/*` (SBML2/3, MATLAB, `MetabolicModel`) | High — already partially tested in `tests/test_models.py` |
| Turbo-Compass | `compass/turbo_compass.py` (`CompassOracle`, `turbo_compass_entry`) | **High** — `MatrixOracle` interface allows a synthetic oracle |
| Turbo-MC (matrix completion) | `compass/turbo_mc/*` (ALS, CV, iterative model, metrics, `matrix_manipulation`) | **High** — pure numpy/sklearn, no solver, no Compass runtime |
| Utilities | `utils.py`, `compass/compass/cache.py`, `globals.py`, arg helpers in `turbo_compass.py` | High |

Existing test coverage: only `tests/test_models.py` (SBML/MAT import + `to_JSON`). Note it is currently **untracked** because `.gitignore` contains `tests/test*` — that pattern must be fixed (e.g. `tests/test*` was almost certainly meant to ignore old `compass/tests/Network*.py` scripts; it also matches `tests/test_models.py`, `conftest.py`, etc.). The legacy `compass/tests/Network1-3.py` files are old dev scripts against a removed API (`compass.run_compass`, `compass._load_cache`) — they are broken and should be deleted or converted.

## 2. The central challenge: runtime

- Compass solves an LP **per reaction per sample**. README: ~30 min/sample for the full RECON2.2 model (7785 reactions).
- The bundled test data is not small: `expression.tsv` is 40,153 genes × 39 cells; the mtx set similar.
- Both bundled "models" are full-size: RECON2.2 (7785 rxn) and iHepatocytes2322 (7930 rxn). There is **no small built-in model** to use for fast end-to-end tests.
- Gurobi requires a license (repo carries a `gurobi.lic` WLS file, but `gurobipy` is not even installed in the local environment; CI would need credentials as a secret).
- Tuning is the point: solver config (`get_gurobi_config` — Method, Presolve, NumericFocus, tolerances), media, and turbo hyperparameters all change results and wall-clock time. Tests must therefore (a) be fast enough to run on every change while tuning, and (b) still provide numeric feedback.

### Strategy: test pyramid with shrinking runtimes

1. **Unit tier (no Gurobi, < 1 min total).** Pure-Python logic. Most of the code that is *cheap* to break (arg manipulation, file I/O, cache format, model transforms, all of turbo_mc) lives here.
2. **Micro-model tier (seconds, real Gurobi).** A hand-built 5-reaction chain model (pattern already prototyped in `compass/tests/Network1.py`) exercised through the real `GurobiOptimizer` and `singleSampleCompass`/cache code paths. Verifies LP construction, delta apply/revert, and end-to-end wiring at tiny scale.
3. **Integration tier (minutes, marked `slow`).** Real model (RECON2.2) but **1 sample + a small reaction subset** via the existing `--single-sample` and `--selected-reactions-for-each-cell` mechanisms. This is the knob that makes the *real* algorithm runnable in ~1–5 min instead of hours.
4. **Turbo tier (synthetic oracle).** The entire turbo machinery is tested against a fake `MatrixOracle` (deterministic low-rank matrix), so `IterativeMCMWithGuaranteedSpearmanR2`, CV models, and the imputation output are validated with zero LP solves. The one real end-to-end turbo test (small subset, `--turbo-max-iters 1–2`) goes in the slow tier.

## 3. Concrete test inventory

### 3.1 `tests/conftest.py` — shared fixtures
- `gurobi_ok()` session fixture: try to construct a trivial Gurobi model and solve; `pytest.skip` if `gurobipy` missing or unlicensed. All LP-dependent tests depend on this.
- `tiny_model()`: programmatic `MetabolicModel` — chain `E1→R1→R2→R3→E4` (from Network1 pattern), with fixed bounds (no `randint`!), one gene per internal reaction. Deterministic expected maxima are analytically known (see Network1.py), which gives us *exact* reference values for the cache/preprocessing tier.
- `tiny_expression(tmp_path)`: small expression TSV with a handful of named genes incl. the ones in `tiny_model`, plus `--single-sample` friendly; avoid loading the 7 MB real TSV in unit tier.
- `media_json(tmp_path)`: minimal media file for the tiny model.
- `slow` marker: `pytest --run-slow` opt-in (via `addopts`/`-m "not slow"` default in CI, or a `RUN_SLOW` env var). Add `pytest-timeout` so no test hangs the suite.
- Session-scoped shared `cache_dir` so the expensive preprocess cache (reaction maxima) is computed **once** and reused across tests within a run — this is the single biggest integration-tier cost (it is exactly what `--precache` exists for).

### 3.2 Unit tier (no solver)
- `tests/test_arg_helpers.py` — `set_argument` / `pop_argument` / `get_argument` in `turbo_compass.py`: missing arg, existing arg, replacement, repeated args, immutability of input for `pop_argument`.
- `tests/test_parse_args.py` — `parseArgs()` with monkeypatched `sys.argv` for: defaults, `--set-license`, `--data-mtx` vs `--data`, turbo arg combinations, `--optimizer` choices, error cases (no data).
- `tests/test_utils.py` — `read_data` on TSV and MTX test data (real bundled files, cheap to read), duplicate-gene handling, anndata path if feasible.
- `tests/test_cache.py` — `compass/compass/cache.py`: write→load round trip, cache-miss raises the documented error, `meta_subsystem` cache dir path.
- `tests/test_metabolic_model.py` — extend existing model tests: `to_JSON` round-trip (parse and compare), `limitExchangeReactions`, `make_unidirectional` (reaction count/direction invariants), species/reaction integrity after import (no orphans in SMAT, bounds sane).
- `tests/test_turbo_mc/` — the highest-ROI new coverage, all solver-free:
  - `matrix_manipulation` functions on small random matrices (known invariants: shapes, idempotence where applicable).
  - `ColumnNormalizerModelWrapper` / `ExcludeConstantColumnsModelWrapper` on synthetic matrices (column means, constant-column removal).
  - `MatrixCompletionFastALS`: convergence on a known low-rank + noise matrix (recovers rank/structure, loss decreases over epochs); determinism with fixed seed.
  - `metrics.py`: Spearman R², etc., against hand-computed values on tiny inputs.
  - `IterativeMCMWithGuaranteedSpearmanR2` with a **synthetic oracle** (subclass `MatrixOracle`; `_observe_entries` returns entries of a known low-rank matrix): assert it terminates, reported CV R² meets `requested_cv_spearman_r2` within tolerance, `impute_all()` approximates the true matrix (correlation > threshold), `max_iterations` cap respected, `plot_progress=False` doesn't error.
- `tests/test_turbo_compass_helpers.py` — the static/file parts of `CompassResourceManager`/oracle that don't run Compass: cache write/validate logic (`_save_cell_names_and_reaction_ids_for_debugging` corruption detection), `_cache_observations` append format, `_load_cache` round trip, `_populate_selected_reactions_file` content for given rows/cols.

### 3.3 Micro-model tier (real Gurobi, seconds)
`tests/test_optimizer.py` (depends on `gurobi_ok`):
- `GurobiOptimizer._build_base_model`: var/constraint counts match model; model solves with zero objective.
- **Delta apply/revert invariant** (this is the crux of correctness for the whole caching scheme): solve with each delta type (`added_uptake`, `added_secretion`, `blocked_reactions`, `high_flux`, objective/sense), then re-solve the *base* problem and assert the objective is identical to the pre-delta base solve — i.e. the model is a "tabula rasa". Also assert var/constr counts return to baseline.
- `maximize_reaction_range` on `tiny_model`: reaction maxima equal the analytic `expected_max` from the chain (exact values, tolerance ~1e-6). This validates the preprocessing/cache values end-to-end with a real solver.
- `singleSampleCompass` on `tiny_model` + `tiny_expression`, `--single-sample 0`, `--num-processes 1`: output `reactions.txt` exists, columns = [sample], all penalties finite and ≥ 0, `success_token` written; second run on same dir is a no-op (cache/skip behavior).
- Blocked/unexpressed reactions: zero-expression sample → reaction penalties of gene-associated reactions take the "inaccessible" value per algorithm (verify against hand computation).

### 3.4 Integration tier (real model, marked `slow`, opt-in)
`tests/test_integration_slow.py`:
- Reuse a session-scoped cache built once (equivalent of `--precache` for RECON2.2 + default media).
- Run `entry()` via subprocess or in-process with `sys.argv` patched:
  `--model RECON2.2 --data <tiny_expression> --single-sample 0 --num-processes 1 --selected-reactions-for-each-cell <subset of ~50 reactions>` → assert output shape, finite penalties, reaction subset honored.
  Budget: after the one-time cache, this should be a few minutes (50 LPs on 1 sample). **This is the test that makes tuning practical**: you can iterate on `get_gurobi_config` (method/presolve/threads) with a bounded, reproducible run instead of a 30-min sample.
- Also a wall-clock **regression/allowance test** in this tier: measure time for a fixed subset and assert it is below 10× the historical baseline. This is an order-of-magnitude smoke check for accidentally doing substantially more work (such as solving every subproblem), not a sensitive benchmark. Baseline timings committed in a JSON fixture; a small `scripts/regen_baselines.py` regenerates them (and regenerates numeric reference outputs below).
- **Golden outputs**: commit reference penalty values for the 50-reaction × 1-sample subset (a small TSV under `tests/reference/`). Test compares within `rtol=1e-4` (or exact if fully deterministic). *Caveat:* solver config changes will legitimately shift values — treat a golden-value failure as "intentional tuning change → regenerate baseline" and document that in the test's docstring. Consider an exact-equality mode for deterministic runs (fixed seed, single thread) since `--num-threads 1` + fixed method should be reproducible on one machine; CI should pin one runner family.
- Determinism test: run the same subset twice (fresh temp dirs) → identical outputs (catches nondeterminism creep, important for turbo which seeds `np.random.seed(1)` but penalty stage should be fully deterministic given the solver config).
- Resume/interruption behavior: partial `_tmp` dir with one sample done → rerun skips completed samples (this is documented user-facing behavior worth pinning).

### 3.5 Turbo end-to-end (in `slow` tier)
- `--turbo <r2> --turbo-increments <dense> --turbo-max-iters 2` on the same small subset: asserts the orchestration (oracle init → iterative fitting → `reactions.tsv` written) completes, output shape correct, turbo cache files (`cell_names.txt`, `reaction_ids.txt`, `turbo_compass_cache.txt`) internally consistent, and cache reuse on rerun.

## 4. Tuning-support features (the "helpful to tune" ask)

1. **Fast numeric loop**: micro-model tests (seconds) catch construction/delta bugs instantly; the 10-reaction timing + 50-reaction golden tests (minutes) give a real-model signal without 30-min waits.
2. **Timing baselines as fixtures** (`tests/reference/timings.json`): per-solver-config wall-clock for the fixed subset. Any `get_gurobi_config` change (Method, Presolve, OptimalityTol, Threads) gets an immediate A/B measurement in CI or locally via `RUN_SLOW=1 pytest tests/test_integration_slow.py -k timing`.
3. **Golden value baselines** with a documented regeneration script, so deliberate tuning changes are a two-command update rather than mystery failures.
4. **Config fingerprinting**: have the timing/golden tests record the active Gurobi config (params + version) in the output artifacts/logs; a mismatched config vs. the fingerprint is reported as a distinct, understandable failure.
5. **Turbo hyperparameters** (`lam`, `n_epochs`, `n_factors`, `train_ratio`) are currently hard-coded in `turbo_compass.py` — while tuning them, the synthetic-oracle tests in 3.2 give sub-second iteration (fit ALS on a 20×20 low-rank matrix, check R²/imputation error) before committing to any real-data run. Consider later refactoring those constants into `parseArgs` options (testable and user-tunable); at minimum, test the model-construction lambdas directly.
6. **Multiprocessing path**: `compass_work` uses `multiprocessing.Pool` + `apply_async`. Add one test with `--num-processes 2` on the micro/tiny data (fork semantics differ on macOS/Windows; CI is Linux, note this) and assert all per-sample outputs appear and `reactions.tsv` aggregates correctly.

## 5. Hygiene / infra fixes to make all of this work

- **Fix `.gitignore`**: `tests/test*` currently ignores the test directory. Replace with targeted ignores (or remove; delete/convert the stale `compass/tests/Network*.py` dev scripts instead of ignoring them).
- **Gurobi in CI**: no license in this env. Options: (a) store WLS credentials as CI secrets and write `gurobi.lic` in the workflow (note: `gurobi.lic` currently sits in the repo root — if it's a real credential, move to git-ignore/secret immediately; it is *not* in `.gitignore`); (b) install `gurobipy` and let the `gurobi_ok` fixture skip LP tests gracefully so the unit tier always runs.
- **Add a test job** to `.github/workflows` (the existing workflow is only a stale-issue closer): PR-triggered job runs the unit+micro tiers (target < 5 min); nightly job runs the `slow` tier with a timeout budget.
- Add `pytest-timeout`, `pytest-mock` to `[project.optional-dependencies] test`.
- `addopts` in `pyproject.toml` already sets `testpaths`; add `-m "not slow"` gating controlled by an env var, and `timeout = 900` for slow tests.

## 6. Suggested order of implementation

1. Hygiene: `.gitignore`, remove/convert `compass/tests/Network*.py`, test-job skeleton in CI with unit tier green (no Gurobi needed).
2. `conftest.py` fixtures (`gurobi_ok`, `tiny_model`, `tiny_expression`).
3. Unit tier tests (arg helpers, parseArgs, utils, cache, turbo_mc, turbo_compass file logic).
4. Micro-model tier (`test_optimizer.py` incl. delta-revert invariant; `singleSampleCompass` on tiny model).
5. Integration slow tier: shared cache fixture, subset run, golden outputs + timing baselines + `scripts/regen_baselines.py`, determinism + resume tests.
6. Turbo: synthetic-oracle tests, then the slow end-to-end turbo test.
7. CI: PR (fast) + nightly (slow) schedule; Gurobi credentials as secret.

Estimated effort: 1–2 days for tiers 1–4, another 1–2 days for integration baselines + CI. Fast tiers keep total runtime well under 5 minutes; slow tier ~10–20 minutes one-time cache + a few minutes per run thereafter.

## 7. Datasets, solvers, and CI licensing (addendum)

### 7.1 Test datasets
- Real fixed data is already committed: `compass/Resources/Test-Data` (39 cells × 40k genes). Use it for one slow-tier test; too heavy for the fast tier.
- Fast tier: generate a small synthetic matrix with a negative-binomial scRNA-seq null model — `counts[g,c] ~ NB(µ_g·s_c, φ)` (φ ≈ 0.1–0.3), CPM-normalized, fixed seed — via committed `scripts/generate_test_data.py`; commit the resulting TSV. Gene symbols must be drawn from the model's gene/isoform associations (genes governing the selected reactions + controls) so the matrix actually drives reactions. Keep ≥ ~16–40 cells so kNN/PCA paths don't degenerate.
- Bound the LP count, not just the data: the full RECON2.2 preprocess cache (7785 reaction-max LPs + metabolite ranges) is ~1h+ and unviable in CI. Either use `--reaction-range` to bound the cache to the subset, or (preferred) commit a reduced SBML submodel under `tests/models/` extracted from RECON2.2 (start from `compass/Resources/Recon2_export/central_carbon_rxns.txt`, 47 central-carbon reactions, expanded to a few hundred with connected reactions). Then even the full cache + full single-sample run is minutes.

### 7.2 Solver strategy (tuning without Gurobi)
- The `Optimizer` abstraction + `--optimizer` flag + `initialize_optimization()` dispatch make a new backend ~150 lines of work. `HighsOptimizer` (HiGHS / `highspy`) is the recommended CI backend: MIT-licensed, no GPU, no credentials, small wheel, and natively supports the delta/revert protocol (change objective, bounds, coefficients, **remove** vars/constraints) — unlike CuOpt, which must rebuild the problem.
- CuOpt is already implemented; not recommended as a CI dependency (NVIDIA licensing is framed around AI Enterprise / developer-program membership, heavy CUDA deps, no GPU on standard runners). Keep as optional third backend.
- Cross-solver consistency test: LP objective values are solver-independent (unique optima), and Compass penalties are driven by objective values. Run the same small subset on HiGHS and Gurobi and assert penalties agree within tolerance — a golden-value-free numerical guard, and the right check while tuning solver configs.
- Note: HiGHS exposes fewer tuning knobs (no method choice; presolve/crossover/tolerances), so timing baselines are HiGHS-based in CI, Gurobi-based locally.

### 7.3 Gurobi in CI via WLS
- The repo-root `gurobi.lic` is in WLS credential format (`key=value`: access id / secret / license id) — the format `utils.parse_gurobi_license_file` already parses. WLS is machine-agnostic (each solve checks in with Gurobi servers), which is exactly what makes it CI-friendly on ephemeral runners.
- CI recipe: `pip install gurobipy` (binaries bundled) → store WLS values as repo secrets → write the license file to `LICENSE_DIR` before pytest. No code changes needed (`utils.get_gurobi_credentials()` picks it up).
- SECURITY: `gurobi.lic` is currently untracked and NOT in `.gitignore`. If it holds real credentials, add it to `.gitignore` immediately and rotate if it was ever committed.
- Academic WLS licenses are renewable/time-boxed: the `gurobi_ok` skip-fixture means a lapsed license degrades to skipped LP tests rather than a red build.

### 7.4 Resulting CI architecture
| Layer | Solver | Data | Runtime | When |
|---|---|---|---|---|
| Unit + turbo_mc + micro-model chain | none / HiGHS | synthetic, committed | < 2 min | every PR, no secrets |
| Integration (submodel or `--reaction-range` subset, 1 sample) | HiGHS | synthetic + bundled Th17 (one test) | ~5–10 min | every PR |
| Cross-solver consistency + timing baselines | HiGHS + Gurobi (WLS secret) | same subset | ~10–20 min | nightly |

## 8. Status (implemented on `test-infrastructure`) and notable findings

### 8.1 What exists now
- `tests/conftest.py`: `build_chain_model()` (analytic 6-reaction chain), `make_chain_expression_df()`, `tiny_model`, `expression_file`, `chain_model_loader`, `compass_args`, `optimizer`/`solver` (parametrized highs/gurobi, skips when unavailable). `solver_available()` probes each backend with a trivial LP.
- `tests/test_optimizer.py`: reaction maxima, uptake/secretion maxima, delta apply/revert "tabula rasa" invariants, infeasibility reporting, and `test_sequential_solves_keep_objective_correct` (see bug below).
- `tests/test_single_sample_compass.py` (micro tier, both solvers): penalty stage vs hand computation (λ=0 ⇒ 1/3 vs 1), `singleSampleCompass` scoring stage vs analytic ground truth (forcing any exchange-limited chain reaction to BETA·r_max forces the whole chain ⇒ score = BETA·(3·p_int + 2·p_E) = 2.85/4.75; R2_neg short-circuits to 0), score ordering, second-run no-op, solver-free penalty stage.
- `tests/test_integration_submodel.py` (slow): end-to-end subprocess runs on the ~190-reaction committed RECON2.2 submodel, cross-solver parity (HiGHS vs Gurobi reaction scores within 1e-6), timing smoke baselines (`tests/reference/timings.json`).
- Measured fast-tier coverage after the micro-tier tests: `compass/compass/algorithm.py` 9%→63%, `penalties.py` 12%→42%, `opt/highs.py` 98%, `opt/gurobi.py` 99%.

### 8.2 Bug found by the tests: highspy name lookup after variable deletion
`highspy` (1.15.1, the MIT wheel used for the license-free HiGHS backend) corrupts name-based column lookup once a model has gone through `addVariable()`/`deleteVariable()` rounds and variables are re-added under previously-used names: `getColByName()` then returns `HighsStatus.kError` **together with an uninitialized index** (observed 32764–32767, i.e. freed memory), and the Python wrapper does not check the status. `HighsOptimizer.solve()` used to resolve reaction names through `getColByName` on every solve, so after the first add/delete round the objective coefficients and bound changes were silently applied to garbage indices — the LPs were solved with the wrong objective/bounds while still reporting "Optimal" (every downstream penalty score silently wrong). This is exactly the silent-failure class the test pyramid exists to catch, and it was invisible to the integration tier alone because the submodel run happened not to trigger the corruption in that one execution path.

Fix (in `compass/opt/highs.py`): variable handles are cached at model build time (base-variable indices are stable because added variables are always appended and deleted in descending index order), and per-solve name resolution uses that cache plus the handles returned by `addVariable()`. No `getColByName` is ever called during `solve()`. `test_sequential_solves_keep_objective_correct` pins analytic objective values across five consecutive solves with added variables and fails on the old code path (HiGHS: 0.0 instead of 1.9).

**Upgrading does not fix it.** The mechanism is in the HiGHS C++ core, not the Python wrapper: `deleteColsInterface()` clears the name hash, `passColName()` only updates it incrementally (leaving it non-empty but incomplete), and `getColByName()` re-forms the hash only when it is empty, then returns an *uninitialized* index on error. highspy 1.15.1 is the latest release on PyPI, and a from-source build of HiGHS master (73cac48, 2026-08-19) still reproduces the bug. Standalone repro: `scripts/repro_highspy_getcolbyname.py` (exit 1 while broken); suite-tracked as `tests/test_upstream_bugs.py` (xfail, flips when upstream fixes it).

**Takeaway:** when driving a solver through a thin wrapper, always check the returned status and prefer index/handle-based access over name-based access on models that are mutated between solves.
