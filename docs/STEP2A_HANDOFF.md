# Step 2A Handoff

## 1. Status

- Step 1 checkpoint commit: `648efd0c81631726afed91e493b143e66e5f25c1`
- Historical Step 2A branch: `feature/d2d-mobo-step2a-core`
- Original milestone state: the implementation was reviewed before the Step 2A
  checkpoint. This handoff preserves that history; publication later uses a
  clean squash containing the sanitized Step 1 through Step 2C tree.
- Overall result: **PASS**
- Production candidate generation: **BLOCKED / NOT RUN**
- Production pull request or candidate release at this milestone: **NOT RUN**

## 2. Summary of implementation

- Added a versioned raw-output-to-utility framework, including nonlinear target
  transforms applied to posterior samples.
- Added integer-index sampling of exact-size, discrete candidate pools without
  materializing the 177.8-billion-row D2D Cartesian product.
- Added R1 UCB-HVI scoring and exact five-candidate, positive-HVI-only batch
  selection.
- Added one sequential, log-space local-penalized selector shared by UCB-HVI and
  qLogNEHVI. It enforces stable ties and hard distances without fallback.
- Added singleton-pool qLogNEHVI with a required configured MC objective and
  pending-point reconstruction at every selection step.
- Added numerical/grid/distance diagnostics and four headless plots per method.
- Extended the read-only workbook auditor to preserve the historical Step 1
  profile and recognize the updated v2 profile.
- Added an explicit production approval gate. The supplied provisional config
  is tested and rejected. The existing campaign proposal runner is also
  blocked before CSV parsing, model fitting, or output creation; even an
  otherwise approved config cannot enter that incompatible legacy path.
- Added a deterministic ten-input synthetic CPU example and optional
  10,000-point scoring benchmark. No workbook is read by either example. The
  synthetic JSON report now records seeds, pool draws/rejections, method
  versions, beta/kappa, MC settings, local penalties, per-selection scores,
  UCB utility moments, pending counts, and runtime versions.

## 3. Mathematical definitions implemented

### Objective transforms

All objective outputs are transformed into an all-maximize utility space.

```text
affine maximize: u(y) = (y - lo) / (hi - lo)
affine minimize: u(y) = (hi - y) / (hi - lo)
Gaussian target: u(y) = exp(-0.5 * ((y - target) / sigma)^2)
negative absolute target: u(y) = -abs(y - target) / scale
```

Affine anchors, target, sigma, and scale are explicit and validated. The
nonlinear transforms are applied to every posterior MC sample before computing
utility mean and population standard deviation (`correction=0`). Latent
posterior sampling is the default and observation-noise inclusion is explicit.

### UCB-HVI

```text
kappa = sqrt(beta)
UCB_j(x) = mean_U,j(x) + kappa * std_U,j(x)
a_UCB-HVI(x) = HV(P union {UCB(x)}; r) - HV(P; r)
```

`P` is the non-dominated observed utility set and `r` is a fixed, caller-supplied
reference in the same utility space. True zero HVI remains zero. Only log-score
stabilization uses epsilon. Negative HVI beyond numerical tolerance raises.
Batch proposal requires enough candidates above a positive-HVI threshold.

### Local penalization

```text
d_w(x,z) = sqrt(sum_j w_j * (x_j - z_j)^2)
phi_i(x) = 1 - exp(-0.5 * (d_w(x,x_i) / rho)^2)
log a_pen(x) = log a_base(x) + sum_i log(max(phi_i(x), epsilon))
```

Distances use normalized input space. Hard selected-to-selected and optional
selected-to-observed/pending thresholds are applied before stable `argmax`.
Exact observed/pending duplicates are always ineligible. An impossible exact
batch raises `UndersizedBatchError` and does not relax settings.

### qLogNEHVI sequential selection

Each remaining candidate is evaluated with shape `N x 1 x D`. At selection step
`t`, `X_pending` contains pre-existing pending points plus selections `1..t-1`.
The seeded qLogNEHVI acquisition is rebuilt and rescored before the shared local
penalty is applied. `ConfiguredMCMultiOutputObjective` is mandatory, preventing
an accidental identity transform in raw outcome space.

## 4. Public APIs

| API | Purpose | Input/output spaces |
|---|---|---|
| `ObjectiveSpec`, `ObjectiveTransform` | Validate and apply a versioned objective contract | raw/model outcome to all-maximize utility |
| `ConfiguredMCMultiOutputObjective` | Use the same transform inside BoTorch MC acquisition | posterior samples to utility samples |
| `sample_discrete_candidate_pool` | Sample exact unique grid tuples with exclusions/constraints | grid indices, physical inputs, normalized inputs |
| `physical_rows_to_grid_indices` | Validate exact grid membership | physical inputs to integer indices |
| `score_ucb_hvi_pool` | MC utility moments and singleton optimistic HVI | normalized pool/raw posterior to utility-space scores |
| `propose_ucb_hvi_batch` | Select an exact positive-HVI batch | candidate pool to selected normalized/physical rows |
| `select_local_penalized_batch` | Shared sequential diversity selector | log acquisition + normalized distances to batch |
| `score_qlognehvi_singletons` | Chunked discrete singleton qLogNEHVI | normalized pool to utility-space log acquisition |
| `propose_qlognehvi_penalized_batch` | Exact sequential qLogNEHVI batch with pending updates | candidate pool to selected normalized/physical rows |
| `summarize_candidate_batch` and plot helpers | Numeric and headless selection diagnostics | normalized/physical inputs to summaries/PNG files |
| `audit_campaign_workbook` | Recognize historical/v2 schemas without saving | workbook cells to structured audit |
| `validate_production_config` | Block unresolved campaign-facing proposals | resolved config to approval receipt/hash |
| `block_legacy_campaign_proposal` | Prevent an approved config from entering the incompatible old proposal runner | resolved config to an explicit Step 2B migration error |

## 5. Files changed

| File | Purpose |
|---|---|
| `configs/d2d_step2a_provisional.yaml` | Deliberately unapproved D2D contract skeleton |
| `docs/D2D_STEP2A_COMPUTATIONAL_CORE.md` | Equations, APIs, determinism, safety, limitations |
| `docs/D2D_OBJECTIVE_CONTRACT_PROVISIONAL.md` | Current workbook facts and unresolved objective choices |
| `docs/D2D_MEETING_DECISION_RECORD_TEMPLATE.md` | Step 2B scientific decision record |
| `docs/STEP2A_HANDOFF.md` | This implementation and verification record |
| `src/mobo_kit/objectives.py` | Objective contract and BoTorch MC adapter |
| `src/mobo_kit/candidate_pool.py` | Integer-index discrete-pool sampling |
| `src/mobo_kit/batch_selection.py` | Shared local-penalized selector |
| `src/mobo_kit/ucb_hvi.py` | Posterior utility moments, HVI, R1 proposal |
| `src/mobo_kit/qlognehvi_batch.py` | Singleton qLogNEHVI and R2 proposal |
| `src/mobo_kit/candidate_diagnostics.py` | Numerical diagnostics and headless plots |
| `src/mobo_kit/workbook_schema.py` | Historical/v2 read-only workbook audit |
| `src/mobo_kit/production_gate.py` | Fail-closed production approval validator |
| `src/mobo_kit/main.py` | Enforce the gate and Step 2A proposal block before campaign data is parsed |
| `src/mobo_kit/cli.py` | Expose the Step 2A proposal-disabled status without pre-creating output |
| `examples/d2d_step2a_synthetic.py` | Synthetic-only ten-input 5/3 smoke run |
| `examples/d2d_step2a_benchmark.py` | Optional synthetic 10,000-point benchmark |
| `tests/test_*.py` (Step 2A modules) | Mathematical, safety, reproducibility, integration tests |

## 6. Tests added

1. Objective identity/affine/target equations, shapes, dtype/device, validation,
   nonlinear sample-before-mean behavior, and BoTorch MC equivalence.
2. Candidate-pool exact size/order, grid membership, normalization, exclusions,
   physical constraints, rejection statistics, impossible requests, and proof
   that Cartesian allocation helpers are not used.
3. UCB-HVI beta/kappa behavior, exact two-/three-objective HVI, Pareto filtering,
   chunk/seed consistency, reference validation, and zero-HVI batch refusal.
4. Local-penalty formula, weights, stable ties, diversity, hard distances,
   duplicate exclusion, and explicit undersized-batch failure.
5. qLogNEHVI singleton shape/chunks, configured objective requirement, reference
   dimensions, observed/pending exclusion, sequential pending counts, exact
   three-candidate batch, and hard-distance failure.
6. Candidate distance/grid/boundary diagnostics and all four headless plots.
7. Historical and v2 sanitized workbook fixtures, aliases, notes, formulas,
   input-grid validation, local file integrity, and unapproved objective status.
8. Production-gate missing/false/nested-placeholder cases; fixed-scaling and
   integer-count validation; distance-weight consistency; rejection of the
   exact supplied provisional config; and pre-I/O campaign-runner blocking.
9. End-to-end CPU synthetic GP fitting, five UCB-HVI candidates, three
   qLogNEHVI candidates, deterministic reruns, spacing, grids, and temporary-only
   outputs.

The suite increased from the protected Step 1 baseline of 79 tests to 170 tests.

## 7. Commands run

```powershell
# Step 1 reproduction before checkpoint
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp <task-work>\step2a-pytest-step1-20260724

# Final full suite
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp <task-work>\pytest-step2a-full-final

# Formatting and dependency checks
.\.venv\Scripts\python.exe -m black --check <changed-and-new-python-files>
.\.venv\Scripts\python.exe -m pip check
git diff --check
git diff --no-index --check NUL <each-untracked-file>
git status --short

# Synthetic verification and optional performance benchmark
.\.venv\Scripts\python.exe examples\d2d_step2a_synthetic.py
.\.venv\Scripts\python.exe examples\d2d_step2a_benchmark.py
```

Focused objective, pool, selector, UCB-HVI, qLogNEHVI, diagnostics, workbook,
gate, and synthetic tests were also run during development.

## 8. Test results

```text
Step 1 checkpoint reproduction: 79 passed, 14 warnings, 18.19 s
Step 2A final full suite:       170 passed, 20 warnings, 7.64 s
Black 25.1.0 check:             PASS
pip check:                      PASS
git diff --check:               PASS
Untracked no-index check:       PASS (22 files)
Private/generated artifact check: PASS (ignored and untracked)
```

Warnings are from the pinned third-party Matplotlib/PyParsing stack and
BoTorch/NumPy array-compatibility path during synthetic GP fitting. There are no
test failures or project-code warnings.

## 9. Performance

- Synthetic smoke elapsed time: 2.72 s for 15 observations, a 128-point R1
  pool, five-candidate UCB-HVI, a 96-point R2 pool, three-candidate qLogNEHVI,
  deterministic reruns, diagnostics, and eight plots on CPU.
- Optional 10k-pool benchmark:
  - pool sampling: 0.1750 s;
  - deterministic three-objective UCB-HVI scoring: 2.2931 s;
  - 6,986 positive-HVI points;
  - full Cartesian grid not materialized.
- Tested environment: Windows 11, CPython 3.12.10, 24 logical processors,
  Torch 2.8.0 CPU (`cuda_available=False`).

| Dependency | Version |
|---|---:|
| NumPy | 2.2.6 |
| pandas | 2.3.1 |
| SciPy | 1.16.0 |
| scikit-learn | 1.7.1 |
| Matplotlib | 3.10.3 |
| seaborn | 0.13.2 |
| PyYAML | 6.0.2 |
| Torch | 2.8.0 |
| GPyTorch | 1.14 |
| BoTorch | 0.15.1 |
| SHAP | 0.48.0 |
| openpyxl | 3.1.5 |
| pytest | 8.4.1 |
| Black | 25.1.0 |

- Observed limitation: singleton hypervolume evaluation scales linearly with
  pool size but has a nontrivial per-point cost; qLogNEHVI construction and GP
  posterior work should continue to use explicit chunks for production-sized
  pools.

## 10. Updated workbook audit

- Input: explicitly supplied ignored private workbook
- Identity and modification time: verified unchanged locally; not tracked
- Sheet/range: `Sheet1`, `A1:AC18`
- Rows/columns: 15 sample rows (Excel 2–16), 29 columns, row 17 blank
- Header mapping: canonical inputs B:K; explicit
  `precur_vol (uL)` to `precur_vol` alias; T uniformity, U optoelectronic, V raw
  thickness, W normalized thickness; Y blank
- No duplicate annealing-temperature field, formulas, or populated objective
  measurements
- Actual note locations: Q18/S18. P18/R18 are blank, contrary to the pack text.
- Grid warning: the local workbook contains an off-grid input discrepancy
  relative to the provisional design contract. No private condition was
  snapped, changed, or copied into tracked fixtures.
- Modification result: file hash and modification time were unchanged by the
  read-only audit and tests; the workbook remains ignored and untracked.

## 11. Production gate

- Required fields include explicit production approval; approved objective and
  constraint statuses; campaign/schema/input
  versions; approver/time/decision record; three unique source/formula/transform
  contracts; fixed scaling and reference point; QC/control/replicate policy;
  explicit constraints list; R1/R2 method, batch, sample, pool, and beta values;
  local radius/hard distances; seed and provenance recording.
- Null, blank, false approval, nested `PENDING`/`TBD`/`provisional`, dynamic
  observed-data scaling, fractional sample/pool counts, inconsistent distance
  weights, invalid numeric, unsupported distance, missing objective, and
  missing policy cases are rejected.
- `allow_hard_distance_relaxation` must be false and R2 sequential pending must
  be true.
- The exact `configs/d2d_step2a_provisional.yaml` template is tested and rejected
  with more than ten independent unresolved reasons.
- `run_mobo_experiment(..., propose_candidates=True)` evaluates the gate before
  CSV parsing or output creation. A fully resolved test config passes the gate
  but is then deliberately rejected because the legacy runner does not use the
  Step 2A transforms, discrete pool, or shared local selector.
- Provisional config status: `approved_for_production: false`.

## 12. Deviations

1. The instruction pack says the workbook notes are at P18/R18. Direct workbook
   inspection and artifact-tool rendering show Q18/S18. The contract fixture
   follows P18/R18; a separate sanitized anomaly fixture and the local audit
   report Q18/S18 without modifying the source file.
2. The pack expects all 15 real workbook recipes to be on-grid. The supplied
   workbook contains a local off-grid discrepancy. The sanitized v2 fixture
   proves valid-grid behavior; the local audit fails closed without exposing,
   snapping, or modifying the private condition.
3. Step 2A was originally reviewed as a feature diff before checkpointing. The
   current public-review tree includes its sanitized implementation in a clean
   Step 1 through Step 2C squash.

## 13. Remaining decisions after the meeting

- Final model/utility mapping: `T,U,W`, `T,U,V`, or another approved mapping.
- Exact T and U ownership, equations, weights, fixed normalization anchors,
  clipping, and missing/failure semantics.
- Final raw thickness versus W model source, 650-nm confirmation, transform,
  sigma/scale, symmetry, and output ownership.
- Fixed utility scales and fixed utility-space reference point.
- R0 row roles, controls, replicates, GP inclusion, QC, failure, outlier, and
  missing-objective policies.
- Complete physical/equipment constraints or explicitly approved `constraints: []`.
- Resolution of the local off-grid discrepancy versus the approved input
  contract.
- R1 beta/posterior samples/pool/seed and R2 MC samples/pool/seed.
- Local-penalty radius, within-batch distance, observed/pending distance, and
  required review plots.
- Workbook/score ownership, campaign state, backup, and audit workflow.

## 14. Recommended Step 2B

- Encode the approved objective mapping and exact versioned formulas.
- Freeze fixed scales and the transformed-utility reference point.
- Encode controls, replicates, QC rules, and every process constraint.
- Resolve the off-grid workbook row or approve a revised input contract.
- Run a read-only dry-run audit on completed R0 objective data.
- Review the resulting model diagnostics and synthetic-equivalent proposal
  audit without writing to the workbook.
- Implement the reviewed Step 2B campaign adapter; do not re-enable the legacy
  raw-objective proposal path.
- Only after the resolved config passes the production gate should a real R1
  proposal be authorized.
