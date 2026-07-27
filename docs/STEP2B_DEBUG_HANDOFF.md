# Step 2B Debug Handoff

## 1. Git status

- Step 1 checkpoint: `648efd0c81631726afed91e493b143e66e5f25c1`
- Step 2A checkpoint: `749c9de` (`Implement D2D MOBO Step 2A computational core`)
- Historical Step 2B branch: `feature/d2d-mobo-step2b-debug`
- Original milestone state: implemented and reviewed locally before the Step 2B
  checkpoint; publication later uses a clean, sanitized Step 1 through Step 2C
  squash.
- Experimental or production pull request at this milestone: not run

Step 2A was reproduced before checkpointing: 170 tests passed, its scoped Black
check passed, `pip check` passed, and `git diff --check` passed.

## 2. Overall result

- Result: **PASS for read-only algorithm debugging**
- Debug R1 proposal: generated
- Experimental approval: **false**
- Production approval: **false**
- Source workbook write-back: not implemented and not run

The generated conditions are not a lab worklist. Every candidate artifact is
watermarked `DEBUG ONLY - NOT APPROVED FOR EXPERIMENT`.

## 3. Source workbook

- Input: explicitly supplied ignored private workbook
- Identity and modification time: verified unchanged locally; not tracked
- Sheet/range: `Sheet1`, `A1:AI20`
- Workbook profile: `d2d_summary_v3_scores`
- Sample rows: 15 unique numeric Sample IDs at Excel rows 2-16
- Note rows excluded: 17-20
- Objective columns: Z, AA, AB
- Workbook modified: no

The previous Step 2A workbook was preserved under an ignored private name. All
`local_inputs/` and `local_outputs/` files remain ignored and untracked.

## 4. Resolved objective contract

Ordered GP outputs and acquisition utilities:

1. Z `Uniformity score`
2. AA `Optoelectronic score`
3. AB `Thickness score`

All three use identity transforms and are maximized. No clipping or observed-data
min/max normalization is applied. GP outcome standardization remains internal to
the existing model implementation. Input normalization uses fixed configured
bounds.

- Fixed utility-space reference: `[-0.01, -10.0, -0.01]`
- All 15 observations strictly dominate the reference component-wise.
- Ignored: AC `Stability score?` and all summary fields AD:AI.

## 5. Score validation

- Required final scores: 15/15 complete and finite for Z, AA, and AB.
- Authoritative Uniformity scores: 15/15 within the required `[0,1]` range;
  out-of-range values now fail validation and model ingestion.
- Uniformity support `L*N*O`: 4/15 match at workbook precision; 11/15 structured
  warnings. Z remains unchanged and authoritative.
- Optoelectronic support `log10(P*Q)`: 15/15 pass against R and AA.
- Thickness support
  `exp(-((mean(valid T1:T4)-650.0)/250.0)^2)`: 15/15 pass against Y and AB.
- Thickness uses the unrounded valid T1:T4 mean, excludes `T anom`, normalizes
  blank/whitespace/NBSP cells to missing, and has no `0.5` exponent factor.
- Missing/non-finite score errors: 0.

The known uniformity discrepancy is recorded in the config, score table, run
manifest, and debug status. It remains a production blocker.

## 6. Control and grid handling

- Control identity: supplied only by the ignored private configuration
- Primary-model inclusion: yes; all 15 R0 observations train the debug GP.
- Measurement provenance assumption: outcomes were measured in the current
  campaign; only the recipe was literature-derived.
- Observed-only exception: supplied only by the ignored private configuration.
- On-grid observed conditions: 14
- Off-grid observed conditions: 1
- Control value changed/snapped: no
- Search grid changed: no
- New candidates all finite, bounded, unique, and exactly on-grid: yes

Strict Step 2A grid conversion still rejects the control. The adapter partitions
the row before grid-index exclusion, but includes all normalized observations in
GP fitting and distance diagnostics.

## 7. GP and R1 debug run

- Training rows: 15
- Model: one CPU `SingleTaskGP` per direct score with internal `Standardize(m=1)`
- Observed Pareto count: 6
- Current fixed-reference hypervolume: `1.709278134536184`
- Candidate pool: 10,000 accepted from 10,000 draws; no duplicate, avoid, or
  constraint rejections in the sampled pool
- Beta/kappa: `4.0 / 2.0`
- Posterior samples: 256
- Local radius: 0.25
- Hard within-batch minimum: 0.15
- Selected unique conditions: exactly 5, each with positive UCB-HVI
- Observed selected-batch minimum normalized distance: `0.8493280824045194`
- Minimum selected-to-observed normalized distance: `0.887172581912801`
- Complete full-study runtime: `73.89 s` on CPU

Training-posterior diagnostics are near-interpolating (`R^2 > 0.99998` for each
objective). These are not cross-validation scores and must not be read as evidence
of out-of-sample accuracy with only 15 observations.

Ignored debug bundle:

```text
local_outputs/d2d_step2b_debug/baseline_seed73/
```

Artifact provenance hashes were recorded in the ignored local audit bundle and
are intentionally not reproduced in tracked documentation.

The tracked handoff deliberately does not reproduce private candidate recipes.

## 8. Replicate worklist

- Unique R1 conditions: 5
- Replicates per condition: 3
- Physical execution rows: 15
- Candidate IDs: `R1-C01` through `R1-C05`
- Replicate numbers: 1, 2, 3
- Inputs are identical within every replicate group.
- Measurement/final-score fields remain blank.
- Replicate aggregation records condition mean, sample standard deviation, count,
  standard error, completeness, and source execution IDs.
- One-row standard deviation/SEM remain missing rather than zero; input mismatch
  inside a group fails.

The synthetic-only R2 boundary aggregates five triplicate R1 groups, combines
them with 15 R0 condition observations (20 total), and returns exactly three
on-grid qLogNEHVI conditions in tests. Its returned metadata is explicitly
synthetic/test-only and not approved for experiment. No real R2 batch was
generated.

## 9. Sensitivity and control ablation

All 13 requested baseline/one-factor/control runs completed without relaxing a
rule, but the debug batch is not robust enough for experimental approval. Pool
randomness and Monte Carlo randomness are independently seeded; the two pool-seed
comparisons below keep the MC seed fixed at 73.

| Comparison | Exact overlap with baseline |
|---|---:|
| beta 1.0 | 5/5 |
| beta 9.0 | 2/5 |
| pool 5,000 | 1/5 |
| pool 20,000 | 4/5 |
| pool seed 137 | 0/5 |
| pool seed 911 | 0/5 |
| radius 0.15 | 5/5 |
| radius 0.35 | 3/5 |
| hard batch distance 0.10 | 5/5 |
| hard batch distance 0.20 | 5/5 |
| posterior samples 128 | 5/5 |
| control excluded | 0/5 |

Control exclusion changed all five selections. Its mean absolute posterior change
on the baseline candidate locations was `0.3748446` across the three raw score
dimensions. The observed Pareto set changed from Samples
`[1, 4, 6, 9, 10, 15]` to `[4, 6, 9, 10, 15]`; the removed control was Pareto
nondominated in the primary fit. Alternative random candidate-pool seeds also
changed all selections. The computational pipeline is working, but the current
15-point campaign fit and finite random pool produce a configuration-sensitive
proposal.

The summary has 13 run-level rows. `sensitivity_candidates_long.csv` records all
65 selected rows with exact inputs, prediction means/stds, raw/log/final
acquisition values, full pairwise-distance rows, nearest-observed distances,
boundaries, settings, and runtimes. `control_ablation.csv` records five aligned
baseline-candidate prediction comparisons and Pareto diagnostics. All are ignored,
watermarked debug artifacts.

## 10. Notebook update

- Added `notebooks/D2D_MOBO_TEST Global Distance Candidate generation.ipynb` from
  the supplied download.
- Inserted `2.5 Calculate and Validate D2D Scores` between sections 2 and 3.
- Uses package score helpers and explicit Z/AA/AB name selection.
- Includes the exact D2D no-half-factor thickness equation.
- Replaced personal paths with repository-relative/configurable paths.
- Replaced unavailable legacy acquisition/model APIs in the active path.
- Retired the undefined legacy global-distance sandbox cells and contradictory
  PCE/stability/log-nEHVI/reference-point guidance.
- Candidate generation is opt-in and routes through the guarded adapter.
- Notebook diagnostics and the guarded candidate bundle use separate output
  directories, so diagnostic plots cannot make the adapter destination nonempty.
- All notebook execution counts and outputs are cleared.
- Programmatic notebook smoke tests pass.

## 11. Tests and quality checks

```text
Step 2A pre-checkpoint:       170 passed, 20 dependency warnings
Step 2B focused suite:         94 passed, 68 dependency warnings
Final complete suite:         251 passed, 74 dependency warnings
Black direct API check:       PASS (14 changed/new Python files)
Python compilation:           PASS
pip check:                    PASS
git diff --check:             PASS
untracked whitespace check:   PASS (13 files)
debug script --help:          PASS
private/generated status:     ignored and untracked
```

Warnings are from the pinned third-party Matplotlib/PyParsing stack and the
BoTorch/NumPy array-compatibility path. No project warning or test failure was
reported.

## 12. Files changed

| File | Purpose |
|---|---|
| `README.md` | Document the guarded Step 2B command and debug boundary. |
| `configs/d2d_step2b_debug.yaml` | Freeze the debug-only v3, objective, control, grid, replicate, and acquisition contract. |
| `src/mobo_kit/workbook_schema.py` | Recognize and audit exact v3 structure plus explicitly configured observed-only exceptions. |
| `src/mobo_kit/candidate_diagnostics.py` | Support visible, metadata-backed debug watermarks on every generated candidate plot. |
| `src/mobo_kit/d2d_scores.py` | Compute and structurally validate D2D support scores without overwriting supplied objectives. |
| `src/mobo_kit/d2d_campaign.py` | Validate config, read workbook rows, prepare control-aware training data, and handle replicates. |
| `src/mobo_kit/d2d_step2b_debug.py` | Run the read-only R1 debug proposal, diagnostics, sensitivity study, and artifact bundle. |
| `src/mobo_kit/d2d_r2_test.py` | Expose only a synthetic-test future R2 qLogNEHVI boundary. |
| `examples/d2d_step2b_debug.py` | Provide the local one-command debug entry point. |
| `notebooks/D2D_MOBO_TEST Global Distance Candidate generation.ipynb` | Add the supplied research notebook with the resolved Section 2.5 and guarded APIs. |
| `tests/test_workbook_schema.py` | Add sanitized v3/profile/immutability/error tests and update the optional local audit. |
| `tests/test_candidate_diagnostics.py` | Verify headless plot generation and embedded debug watermark metadata. |
| `tests/test_d2d_scores.py` | Test equations, missing semantics, tolerances, and severity policy. |
| `tests/test_d2d_campaign.py` | Test the pinned config/grid, objective order, control partition, row metadata, strict triplicates, and 20-condition aggregation. |
| `tests/test_d2d_step2b_debug.py` | Test fail-before-output, safe output containment, independent seeds, deterministic batch, detailed artifacts, watermarks, and workbook integrity. |
| `tests/test_d2d_r2_test.py` | Test synthetic triplicate aggregation, pool-coordinate consistency, test-only status, and exact three-condition qLogNEHVI. |
| `tests/test_d2d_notebook.py` | Test executable thickness helpers, section order, resolved narrative, active APIs, portability, and cleared output. |
| `docs/STEP2B_DEBUG_HANDOFF.md` | This handoff and production-blocker record. |

## 13. Deviations and local issues

1. The pack allowed the reference point to be chosen; its resolved
   `[-0.01, -10.0, -0.01]` value was used exactly.
2. The workbook contains formulas only in R2:R3; R4:R16 are cached/static numeric
   checks. Validation accepts either formulas with numeric cached values or static
   numeric checks.
3. The supplied notebook required more than a single inserted cell because its
   active path referenced unavailable APIs, undefined variables, and personal
   paths. The research notes were retained where safe, but the legacy sandbox was
   visibly retired.
4. Black's normal Windows worker CLI left non-responsive worker processes after
   reporting completion. The final formatting/check used Black's direct serial
   API, passed, and the task-owned stale workers were terminated.

## 14. Remaining blockers before experimental R1

- Correct or explicitly approve the 11 inconsistent uniformity scores.
- Confirm the configured control's measurement provenance through the private
  campaign record.
- Review the strong control-ablation and random-pool-seed sensitivity.
- Choose and freeze a reviewed pool construction/seed and acquisition configuration.
- Review the five-condition proposal with experimental and computational owners.
- Approve a production provenance/config record; both approval flags remain false.
- Define the final workbook write-back/interface and signing/trust policy.

## 15. Recommended next step

Hold a bounded computational/experimental review of control provenance and the
seed/pool sensitivity. After resolving the uniformity values, freeze the reviewed
configuration and rerun one production-gated dry run for sign-off. Do not fabricate
from the current debug bundle.
