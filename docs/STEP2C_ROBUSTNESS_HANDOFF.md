# Step 2C Robustness Handoff

## 1. Baseline and review status

- Step 1 checkpoint: `648efd0c81631726afed91e493b143e66e5f25c1`
- Step 2A checkpoint: `749c9de` (`Implement D2D MOBO Step 2A computational core`)
- Step 2B checkpoint: `1c6a83a9dfd7e6ed5e69ce66765b8dc0fd8a86af`
- Step 2C implementation and read-only robustness evidence: complete for review

## 2. Overall result

- Implementation and read-only robustness study: **PASS**
- Full stable consensus criteria: **FAIL**
- Experimental approval: **false**
- Production approval: **false**
- Real R2 proposal: not run

Step 2C completed the declared full study and produced a validated robust-region
bundle. It did not release an R1 consensus batch. This is the correct fail-closed
outcome: search convergence was strong, but only four regions met the required
study-family coverage and the GP validation remains poor.

## 3. Source workbook

- Input class: Git-ignored private campaign workbook
- Content identity and unchanged-file proof: verified before, during, after, and
  at independent validation; exact identifiers are intentionally omitted
- Adapter/schema contract: verified locally; private workbook metadata is not
  tracked
- Workbook modified: no
- Known Uniformity mismatch: retained as warning-only input for debugging and as
  an experimental-approval blocker
- Control observation: retained in the primary model under the reviewed
  off-grid policy; its exact override remains in the ignored private config

No code path saves or writes back to this workbook.

## 4. Objective and baseline contract

The direct supplied scores in Z/AA/AB remain ordered as Uniformity,
Optoelectronic, and Thickness. All three use identity transforms and are
maximized. The fixed utility reference is `[-0.01, -10.0, -0.01]`.

Step 2B was checkpointed before Step 2C. Step 2C intentionally replaces the
finite random-pool/MC-moment debug search with analytic identity moments, exact
nested Sobol prefixes, and deterministic grid refinement. The control inclusion,
score-source, input-grid, reference-point, and approval boundaries did not
change.

## 5. Analytic identity moments

- Primary API: `posterior_identity_moments`
- Full diagnostic comparison: 2,048 MC samples, seed 73, 2,048 candidates
- Maximum absolute mean difference: `0.00006279358632210741`
- Maximum absolute standard-deviation difference: `0.00036946014787431203`
- Exact selected overlap: 5/5
- Regional matches within 0.15: 5/5
- Analytic/MC debug gate: pass
- Observed runtime ratio, MC/analytic: `1.3323`
- Production Step 2C moment method: deterministic `analytic_identity`

The per-objective comparison tolerance is
`max(0.02, 0.02 * max(observed_range, 1.0))`. Monte Carlo remains diagnostic;
it does not drive the full search.

## 6. Nested Sobol pools

Primary accepted prefixes were exact and nested:

| Seed | Accepted size | Draws | Duplicate/avoid/constraint rejects | Prefix SHA-256 |
|---:|---:|---:|---:|---|
| 73 | 16,384 | 16,384 | 0/0/0 | `76B069D765949EDF045E8ACDBA0866EB27125FF84CB2AC8A3F874E38B9CBC3B2` |
| 73 | 32,768 | 32,768 | 0/0/0 | `896A8483D182BEC14584149F657E7C9EADE07C3207A4CAB28F877EB7E3CE6833` |
| 73 | 65,536 | 65,536 | 0/0/0 | `84ABB5185246D20336CB90BD1E32836E8C0EDCF73AA5FAF288353D97770B9EB7` |
| 73 | 131,072 | 131,072 | 0/0/0 | `848B418477DEF6DF9FF04287F4FA902AF8FE15490AFD6AC2801069799FA33DBC` |
| 137 | 131,072 | 131,072 | 0/0/0 | `4AEE3ECE6A3F181B4D0A1E7A1A4F6F6E4613C4C2B52E258DCC5820CB901665D9` |
| 911 | 131,072 | 131,072 | 0/0/0 | `F8CC95A722C309FB1B1BB21C62B1F8DC9AFBBD60E8227FB283E15C0B52BE9BD2` |

The full Cartesian grid was never materialized. Observed on-grid recipes were
excluded by exact grid index; the off-grid control remained a continuous GP and
distance reference only.

## 7. Local refinement and convergence

- Full settings: 64 anchors per selection step, at most 10 sweeps,
  `1e-10` improvement tolerance, all allowed coordinate values
- Mean nested-batch acquisition gain: `1.73963296072043`
- Gain range across the four prefixes: `1.6522526527608` to
  `1.86429022721892`
- Baseline anchor summaries: 358
- Distinct baseline converged optima: 21
- Accepted baseline coordinate moves: 2,845
- Study candidate rows: 75
- Invalid grid, bounds, hard-distance, or duplicate rows: 0

All adjacent nested-prefix comparisons from 16,384 onward matched 5/5 within
0.15 with mean matched distance 0.0. Both secondary scramble seeds reproduced
the same five refined optima exactly; their acquisition regrets were numerical
roundoff (`-6.1e-14` and `2.4e-13`). Refinement is deterministic in regression
tests and never accepts a score decrease.

## 8. Model validation

Exact leave-one-out results for all 15 observations:

| Variant | Objective | MAE | RMSE | R2 | 68% coverage | 95% coverage | Mean NLPD | Max abs. standardized residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| default | Uniformity | 0.2062 | 0.2472 | -0.9102 | 0.400 | 0.533 | 4.096 | 9.420 |
| default | Optoelectronic | 0.3853 | 0.5266 | -0.2473 | 0.467 | 0.667 | 2.821 | 6.571 |
| default | Thickness | 0.3559 | 0.4299 | -0.4578 | 0.467 | 0.600 | 1.354 | 3.935 |
| conservative | Uniformity | 0.2045 | 0.2478 | -0.9188 | 0.467 | 0.667 | 2.384 | 7.851 |
| conservative | Optoelectronic | 0.3988 | 0.5414 | -0.3185 | 0.467 | 0.667 | 3.088 | 6.384 |
| conservative | Thickness | 0.3604 | 0.4328 | -0.4779 | 0.467 | 0.600 | 1.316 | 3.804 |

Every one of the 96 full/fold objective fits had at least one normalized ARD
lengthscale at or above the declared flatness threshold of 10.0. One fit also
had a lengthscale at or below 0.05. Flat flags were most frequent for `time_2`
(90/96), `precur_conc` (89/96), `anneal_time` (87/96), `anti_vol` (86/96), and
`anti_time` (83/96).

All 96 fits also reached their configured likelihood-noise lower bound: all 48
default fits were at `0.001`, and all 48 conservative fits were at `0.01`. This
is a major calibration diagnostic, not evidence that either noise policy is
validated.

The 576 recorded optimizer warnings are the same NumPy `copy=` deprecation
warning, not fit failures. During the full influence study, the console also
showed one GPyTorch numerical warning that clamped a tiny negative posterior
variance to `1e-10`. That console-observed warning was not suppressed, but it is
not present in the aggregate warning CSV.

For both model variants, all five selected Optoelectronic posterior means were
above the observed maximum. No selected posterior mean violated the declared
Uniformity or Thickness bounds. These diagnostics reinforce that the models are
not ready to justify fabrication.

## 9. Observation influence

All 15 exact omission runs completed on the common accepted pool. The five most
influential observations were Samples 1, 9, 4, 6, and 10.

Sample 1 was rank 1/15 at the 100th percentile. Omitting it produced 0/5 exact
or 0.15-regional matches, mean matched batch distance `2.1692`, prediction-mean
change `0.6183`, normalized acquisition-rank change `0.3345`, and mean absolute
hyperparameter log-ratio `2.8595`. Its omission removed Sample 1 from the
observed Pareto set (Pareto Jaccard `0.8333`). The control remains included in
the primary model; this result is a sensitivity warning, not an automatic
exclusion decision.

## 10. Bounded-utility and qLogNEHVI compatibility

The declared clip-UCB and unbounded-UCB policies had 0/5 exact and 0/5 regional
correspondence. The clip policy affected 77,286 of 131,072 pool rows (58.965%)
and all five selected rows; the largest selected-coordinate clip was `0.04911`.
The bounded and unbounded acquisition sums were `7.6932` and `13.2048`,
respectively. These policy-specific HVI sums should not be interpreted as a
shared-scale regret.

Training targets were not mutated. A real synthetic BoTorch qLogNEHVI test now
uses the same bounded posterior-sample objective with a fixed reference and
seed. No real R2 qLogNEHVI proposal was generated.

## 11. Local-penalty study

| Variant | Comparator | Exact/regional overlap | Minimum batch distance | Acquisition sacrifice | Interpretation |
|---|---|---:|---:|---:|---|
| no soft, no hard spacing | self | 5/5 | 0.2828 | 0 | Base optima already separated |
| no soft, hard 0.15 | no soft, no hard | 5/5 | 0.2828 | 0 | Hard spacing inactive |
| radius 0.15 | no soft, hard 0.15 | 4/5 | 0.5657 | `2.21e-7` | Material diversity change |
| radius 0.25 | no soft, hard 0.15 | 3/5 | 0.7141 | 0.00378 | Material diversity change |
| radius 0.35 | no soft, hard 0.15 | 3/5 | 1.0000 | 0.11789 | Material diversity change |

The primary radius 0.25 is active in the converged full search even though its
mean penalty factor remains close to one (`0.9865`). The classification uses
regional and pairwise-distance changes, not score attenuation alone. Hard
spacing did not relax.

## 12. Beta, model, and boundary robustness

Beta 1 and beta 9 each had 0/5 correspondence with beta 4 after refinement. The
conservative model retained 4/5 exact clustered regions, whereas unbounded UCB
retained 0/5.

The baseline batch remains strongly boundary-seeking:

- `speed_2`: 5/5 at the upper bound;
- `precur_conc` and `precur_vol`: 5/5 at the upper bound;
- `anneal_temp`, `anti_vol`, and `anti_time`: 5/5 at the lower bound;
- `time_2` and `anneal_time`: 2/5 at each lower and upper bound.

Lower and upper rates and enrichment are reported separately. The combination
of beta sensitivity, bound-policy sensitivity, flat ARD directions, and boundary
seeking remains a fabrication blocker.

## 13. Robust regions and shortlist

- Clustering: deterministic complete-link agglomerative clustering
- Primary normalized distance threshold: 0.15
- Sensitivity thresholds: 0.10 and 0.20
- Robust regions: 18
- Debug shortlist: 12 medoids
- Regions covering one, two, and four non-baseline families: 9, 5, and 4
- Regions meeting the required at-least-three-family criterion: 4

The tracked handoff contains no private recipe rows. Those remain only in the
ignored local artifacts.

## 14. Conditional consensus result

- `r1_consensus_debug_batch.csv`: not created
- Full-mode eligibility: pass
- Largest-two nested regional match gate: pass, 5/5
- Largest-two mean-distance gate: pass, 0.0
- Five family-qualified regions: fail, only four available
- Exact five consensus candidates: fail, only four eligible medoids

All candidate-row gates are consequently false because no exact five-row set
exists; this does not mean the global debug/approval flags were weakened. Both
approval flags remain false throughout. The explicit reason artifact is
`r1_no_stable_batch_reason.json`.

## 15. Local and public artifacts

Full campaign evidence remains below the ignored `local_outputs/` boundary and
must not be committed or shared. Private runs label their complete archive
`*_PRIVATE_EVIDENCE_DO_NOT_SHARE.zip`. The sibling `<run>.zip` is now a separate,
strictly allowlisted public summary containing aggregate tables and sanitized
metadata only. Input-specific artifact hashes are intentionally not tracked.

## 16. Verification

- Sanitized synthetic end-to-end suite: 2 passed, including the real workbook
  adapter, strict GP/search orchestration, resolved-config provenance,
  rollback-safe publication, and validation of the aggregate-only public ZIP
- Ignored private Step 2B/2C configuration compatibility smoke: pass
- Historical pre-sanitization private fast/full runs: pass with 46 validated
  artifacts; these private bundles were not published
- Historical full-mode runtime: `1508.32 s` on CPU
- Complete pytest: 427 passed, 2 expected private-input skips, 270 dependency
  warnings (`48.67 s` with single-threaded numerical libraries)
- Black: pass on all 67 changed/new Python files
- Python compilation: pass
- `pip check`: pass (`No broken requirements found`)
- `git diff --check`: pass
- Private/generated outputs: confirmed ignored by `.gitignore`

Largest full-mode phase runtimes were observation influence `707.34 s`,
secondary/model/bound/beta studies `413.08 s`, primary moments and scores
`285.17 s`, and nested refinement `64.39 s`.

## 17. Files changed

| Area | Files/purpose |
|---|---|
| Config/docs | `configs/d2d_step2c_debug.yaml`, this handoff, the Step 2C method document, and README command/safety guidance |
| Entry points | Private fast/full runner and separate sanitized synthetic-CI example |
| Search | Analytic UCB-HVI moments, nested Sobol prefixes, regional batch comparison, and deterministic discrete refinement |
| Models | Strict GP variants, exact LOOCV, hyperparameter/flatness diagnostics, and all-observation influence |
| Policies | Bounded UCB, bounded qLogNEHVI objective compatibility, beta and five-variant penalty studies |
| Stabilization | Complete-link robust regions, diverse shortlist, exact consensus gates, and lower/upper boundary diagnostics |
| Safety | Strict artifact schemas/provenance, CSV evidence-chain validation, watermark and format checks, exact consensus/no-stable linkage, aggregate-only public ZIP validation, and rollback-safe public/private publication |
| Tests | Unit, regression, artifact-negative, real qLogNEHVI, pending-row, normalization, plotting, and sanitized end-to-end coverage |

## 18. Deviations and limitations

1. A separate sanitized synthetic runner was added because campaign fast mode is
   correctly pinned to the private workbook and therefore cannot be portable CI.
2. The qLogNEHVI work is compatibility testing only; Step 2C deliberately did
   not generate a real R2 proposal.
3. Region-threshold sensitivity exports counts at 0.10 and 0.20; full membership
   rows are exported for the declared primary 0.15 threshold.
4. One console-observed posterior-variance clamp warning occurred in the full
   study and is retained as a model-stability limitation; it was not captured in
   the aggregate warning CSV.

## 19. Remaining blockers before experimental R1

- Correct or explicitly approve the 11 inconsistent supplied Uniformity scores.
- Confirm Sample 1 outcome provenance with the experimental team.
- Resolve poor LOOCV calibration/accuracy, all 96 likelihood-noise fits reaching
  their configured floor, and the pervasive flat ARD directions.
- Review the posterior-variance numerical warning.
- Choose and justify the bounded-versus-unbounded utility policy.
- Review the strong beta, control, and boundary sensitivity.
- Obtain at least five regions satisfying the declared family-coverage gate, or
  explicitly redesign that gate through a new reviewed specification.
- Freeze a production configuration and approval record.
- Define workbook write-back/interface and signing policy only after candidate
  sign-off.

## 20. Recommended next step

Do not fabricate from either Step 2B or Step 2C artifacts. First correct/approve
the Uniformity scores and confirm the control provenance. Then perform a focused
model review—kernel/priors, noise treatment, objective policy, and whether more
R0 information is required—before rerunning this same fail-closed full study.
Proceed to Excel button/write-back integration only after a five-row consensus
set and experimental sign-off both exist.
