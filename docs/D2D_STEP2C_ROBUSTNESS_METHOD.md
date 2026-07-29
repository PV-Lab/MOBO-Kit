# D2D Step 2C: R1 Robustness, Search Convergence, and Proposal Stabilization

## Status and safety boundary

Step 2C is a read-only computational audit. It does not approve fabrication,
change the campaign workbook, generate a real R2 proposal, or change the Sample
1 inclusion policy. All outputs carry:

```text
DEBUG ONLY - NOT APPROVED FOR EXPERIMENT
```

The campaign runner accepts only the pinned private workbook and fail-closed
debug configuration. It verifies the workbook SHA-256 and modification
timestamp before computation, after computation, and after artifact export.
Output is limited to the configured Git-ignored directory and is published from
a fully validated staging directory. A separate synthetic-CI helper generates a
sanitized workbook under the ignored output root and exercises the same internal
orchestration without making the campaign command redirectable.

The known supplied-Uniformity mismatch and the off-grid control exception remain
visible and continue to block experimental approval.

## Objective contract

The three supplied final scores in columns Z, AA, and AB are used directly, in
this order:

1. Uniformity score;
2. Optoelectronic score;
3. Thickness score.

All three are maximized. The objective transform is identity for Step 2C. The
declared UCB-support policy clips only Uniformity and Thickness UCB coordinates
to `[0, 1]`; it never clips training targets or the Optoelectronic score.

## GP validation

Two explicit model variants are evaluated:

- `dim_scaled_prior`, matching the Step 2B model;
- `conservative`, with documented observation-noise and ARD-lengthscale floors.

Every fit is strict: failed optimization raises rather than silently returning
an unfitted model. Exact leave-one-out validation fits 15 folds per variant and
reports per-observation predictive means, predictive uncertainty, residuals,
standardized residuals, interval inclusion, and objective-level accuracy and
calibration metrics. Full-fit and fold hyperparameters, optimizer warnings, and
leave-one-out hyperparameter stability summaries are exported separately. ARD
diagnostics flag normalized lengthscales at or below 0.05 as very small and at
or above 10.0 as extremely large/flat. Candidate posterior means outside the
observed range and outside declared objective bounds are reported separately.

Training-posterior diagnostics are not substituted for leave-one-out results.

## Analytic identity moments and UCB-HVI

Because every Step 2C objective transform is identity, posterior mean and
standard deviation are obtained analytically from the GP posterior. The primary
search is therefore deterministic and does not depend on a Monte Carlo seed.

A fixed-seed Monte Carlo comparison remains in the audit. It records moment
differences, selection correspondence, runtime, per-objective tolerances of
`max(0.02, 0.02 * max(observed_range, 1.0))`, and pass/fail flags. This
comparison is diagnostic and does not relax any consensus gate.

Candidate utility is computed as:

```text
UCB = posterior mean + sqrt(beta) * posterior standard deviation
```

followed by the selected UCB-bound policy and deterministic hypervolume
improvement relative to the declared reference point.

## Nested Sobol search and exact-grid refinement

Full mode uses exact accepted-unique scrambled Sobol prefixes of 16,384,
32,768, 65,536, and 131,072 points with primary seed 73. Every smaller accepted
pool is an exact prefix of the next. Full-size secondary scramble seeds 137 and
911 measure scramble sensitivity. Pool hashes and rejection counters prove the
search basis without materializing the full Cartesian grid.

At each sequential batch step, the highest eligible pool anchors and eligible
previously discovered optima are refined by deterministic coordinate ascent.
Every allowed value of one grid dimension is evaluated at a time. Stable
lexicographic tie-breaking, positive-HVI eligibility, observed-row exclusion,
hard spacing, soft penalty, sweep limits, and termination reasons are recorded.
The trace includes anchor pool indices, start/end base HVI, start/end penalized
log score, accepted moves, changed dimensions, and aggregate sweeps.

Each nested size exports both the unrefined pool batch and refined batch,
including refinement gain, HVI summaries, boundary counts, matched distances,
relative regret, and runtime.

## One-factor robustness studies

The core candidate-region registry contains 13 unique runs spanning:

- four nested-search sizes;
- two alternate Sobol scrambles plus the primary reference;
- default and conservative models;
- unclipped and clipped UCB policies;
- beta values 1, 4, and 9;
- no-soft and three soft-radius penalty settings under fixed hard spacing.

The primary run acts as the shared reference level. Family-level persistence is
weighted equally so families with more variants do not dominate.

The local-penalty table separates the hard-spacing effect from the soft-penalty
effect: soft-radius variants are compared with the no-soft run that retains the
same hard spacing. Acquisition sacrifice, diversity, penalty factors, boundary
behavior, nearest-observed distance, runtime, and region change are retained.
Its human-readable activity classification considers regional correspondence
and minimum/mean pairwise-distance changes, not only score attenuation. The
overall soft-penalty interpretation is derived from `radius_*` variants only;
hard spacing is interpreted separately.

## Observation influence

The full model and every exact leave-one-out model are evaluated on the same
accepted pool and use the same local-refinement settings. The report keeps the
following components visible:

- exact and regional batch displacement;
- prediction changes at the full-model batch and robust-region medoids;
- common-pool acquisition rank and top-K changes;
- observed Pareto membership changes;
- hyperparameter displacement;
- fit/proposal runtime and fitting-warning counts.

The composite influence rank is a summary, not a replacement for these
components. Sample 1 remains included in the primary model regardless of its
diagnostic rank.

## Robust regions and shortlist

Core candidates are clustered in normalized input space with deterministic
agglomerative complete linkage. The primary distance threshold is 0.15, with
0.10 and 0.20 sensitivity counts. Each region exposes its medoid, diameter,
run/family coverage, equal-weight study-family persistence, within-run normalized
HVI statistics, prediction summaries, nearest-observed distance, boundary
frequency, and full-versus-omit-Sample-1 correspondence.

The ignored robust shortlist contains 8-12 diverse medoids where geometry
permits. It includes predictions and uncertainty under both models, raw and
bounded UCB coordinates, observed-range flags, boundary flags, and omission
sensitivity at each medoid.

Lower- and upper-boundary counts, rates, and enrichment are kept separate in
candidate, influence, shortlist, and plotting artifacts. This prevents opposite
boundary tendencies from cancelling in a combined statistic.

## Future qLogNEHVI compatibility

Step 2C does not generate a real R2 batch. The bounded posterior-sample objective
is nevertheless integrated with the singleton qLogNEHVI scorer and tested on a
small real BoTorch model with a fixed reference point and seed. Bounds apply only
to posterior utility samples; training targets remain unchanged.

## Consensus gate

A five-row `r1_consensus_debug_batch.csv` can be created only in full mode and
only when all declared checks pass on those exact five rows:

1. the two largest nested searches match at least 4/5 regions within 0.15;
2. their mean matched distance is at most 0.10;
3. at least five regions, including every chosen region, cover at least three
   non-baseline core study families;
4. the chosen medoids are finite, bounded, unique, exactly on-grid, and satisfy
   the 0.15 hard pairwise distance;
5. debug-only and both approval-false flags are intact.

Fast mode is categorically ineligible. If any check fails, the runner creates
`r1_no_stable_batch_reason.json` and does not create a consensus batch.

Before publication, the validator independently checks required table schemas,
provenance fields, the canonical resolved/source config relationship, workbook
proofs, exact conditional artifacts, debug stamps, PNG metadata, and supported
file formats. It recomputes the convergence and family gates from their CSVs,
links robust regions to the shortlist and the shortlist to any consensus rows,
and requires the no-stable reason to reproduce the failed checks and observations
exactly. The ignored output directory remains the complete local evidence
surface. For private campaign runs, its full ZIP is explicitly named
`*_PRIVATE_EVIDENCE_DO_NOT_SHARE.zip` and carries a warning because it contains
local provenance, sample-level data, and exact candidate recipes. The legacy
`<run>.zip` sibling is now a separate, strict public-summary export: it contains
only allowlisted aggregate tables plus a sanitized manifest and excludes local
paths, profile names, workbook filenames, sample-level rows, recipes, and
recipe-coordinate plots. The directory and both requested ZIPs are published in
one rollback-safe transaction; prior artifacts are restored if publication or
either final validation fails.

## Running the audit

From the repository root:

```bash
# Sanitized generated-data end-to-end CI coverage
python examples/d2d_step2c_synthetic_ci.py --overwrite

python examples/d2d_step2c_robustness.py --mode fast
python examples/d2d_step2c_robustness.py --mode full
```

The synthetic command is the self-contained generated-data end-to-end CI path.
Campaign fast mode is a private-workbook integration smoke only. Full mode is the
declared robustness study. Every mode is ignored, read-only with respect to its
source workbook, and debug-only.
