# Campaign status and how to use it

Snapshot for collaborators. The full loop runs: R0 LHS -> R1 UCB-HVI (5) ->
R2 qLogNEHVI (3), three replicate films per condition, 23 distinct conditions.

## Running a round

```python
from mobo_kit.campaign import load_campaign_config, run_r0_lhs, run_r1_ucb, run_r2_qlognehvi

config = load_campaign_config("configs/campaign_d2d_perovskite.yaml")

r0 = run_r0_lhs(config, n=15)                        # space-filling, no model
r1 = run_r1_ucb(config, X_phys, Y_model, n=5)        # UCB-HVI + local penalisation
r2 = run_r2_qlognehvi(config, X_phys, Y_model, n=3)  # qLogNEHVI
```

Each returns a `RoundResult` with:

| field | contents |
|---|---|
| `conditions` | distinct proposed conditions, physical units, columns = input names |
| `replicates` | one row per film, with `candidate_id` / `replicate_group` / `replicate_index` |
| `diagnostics` | method, seed, pool size, objective contract version, validity report |

`diagnostics["validity"]` carries `min_pairwise_distance` and
`boundary_coords_per_condition`, which are the numbers to plot per round.

## What `Y_model` must contain

**Not the three stored score columns.** Since 2026-07-30 the objectives are
computed in Python from the raw measurement columns, and the stored cells are a
cross-check. Column order comes from `objective_names(config)`:

```
("uniformity", "optoelectronic", "thickness")
```

`read_campaign_workbook` returns exactly that as `contents.model_values`, so the
normal path is:

```python
from mobo_kit.workbook_io import read_campaign_workbook

contents = read_campaign_workbook("local_inputs/Summary Table.xlsx", config)
X_phys = contents.inputs.to_numpy(float)
Y_model = contents.model_values.to_numpy(float)     # objective order
assert contents.errors == ()                        # fail closed before fitting
```

Each value comes from a recipe declared in config (`objectives.specs[].measurement`):

| objective | recipe | from |
|---|---|---|
| uniformity | `product` | `Coverage`, `1 - Uniformity` (computed), `Phase purity` |
| optoelectronic | `log10_product` | `PL - Implied Voc (Max)`, `Photoconductance (Max)` |
| thickness | `mean_of_present` | whichever of `T1..T4` were measured |

Thickness is in **nanometres**, unrounded, because the GP trains on the raw
measurement and the 650 nm Gaussian is applied to the posterior. See
`GP_MODEL_DECISION.md` for why. Anything that collects data for the next round
must collect nm.

`contents.findings` carries what the read noticed: cross-check mismatches,
readings the operator excluded, and films whose thickness readings disagree.
`contents.errors` is empty on the R0 rows; if it ever is not, do not fit.
`contents.inputs_used` records how many readings each value came from, which is
what Phase 4 needs to turn a spread into an observation variance.

## For the plotting work

**Contour slice through the GP.** Fit with the same path a round uses, then
evaluate on a 2-D grid with the other eight inputs held fixed:

```python
from mobo_kit.campaign import (
    build_objective_transform,
    fit_campaign_models,
    normalise_inputs,
)

# same normalisation, structured means, variant and seeding as the round itself,
# so this reproduces the round's model rather than a similar one
model = fit_campaign_models(config, X_phys, Y_model, seed=73)

model.eval()
with torch.no_grad():
    post = model.posterior(torch.tensor(grid_norm))   # grid_norm in [0,1]^10
    mean, var = post.mean, post.variance
```

Two things to respect when turning that into a utility surface:

* the GP output for thickness is **log(nm)**, not nm. `ObjectiveSpec.model_link`
  records this. Use `transform.expected_transform(mean, var)` rather than
  transforming the mean yourself; it dispatches per objective and integrates the
  lognormal by quadrature where needed.
* inputs are normalised to `[0,1]` against the config grid bounds, not the
  observed range. `normalise_inputs(config, X_phys)` is the conversion. A model
  fitted on config bounds and evaluated on observed-range coordinates is being
  asked about different points than it was told about, and nothing errors.

**Round-comparison plot.** Keep each `RoundResult` and plot `conditions` per
round on shared axes (R0 grey / R1 blue / R2 orange), plus per-round
`min_pairwise_distance` and boundary counts from `diagnostics`. Contour slices
should show **23 distinct conditions**, not 39 films -- replicates share inputs
and would otherwise overplot.

Hypervolume is comparable across rounds only because objective scales are fixed
in config; `assert_scaling_is_campaign_fixed` enforces that. Do not re-derive
scales from observed data between rounds.

## Model state

Validated on the 15 R0 observations, exact leave-one-out, null R2 = -0.148:

| objective | plain GP | with structured mean |
|---|---:|---:|
| thickness (nm) | +0.183 | **+0.384** |
| optoelectronic | -0.342 | **+0.267 to +0.355** (see open issues) |
| uniformity | no learnable signal (permutation p = 0.82) | n/a |

Uniformity is exploration-only by measurement, not by choice. The interface must
not imply the model knows more than it does about it.

## Reading a round's results back

`read_candidate_results(source_workbook, config, "R1")` reads the filled-in
candidate sheet and returns design points, not films:

| field | contents |
|---|---|
| `conditions` | one row per condition, input columns |
| `model_values` | one row per condition, objective columns, **aggregated** |
| `replicates` | one row per film, with its own objective values |
| `replicate_spread` | per-condition sd, in each objective's aggregation space |
| `films_used` | how many films each observation was aggregated from |
| `findings` | the same note / warning / error list as the source read |

Objective values are computed per film with the same recipes Sheet1 uses, so R0
and R1 observations are commensurable, and only then aggregated per
`replicate_group`.

**Thickness aggregates in log space** (`replicate_aggregate: mean_of_log`), because
`response: log` means the GP trains on `log T` — the geometric mean is the
arithmetic mean in the space the model works in, and it is the choice consistent
with pooling `train_Yvar` in log space. The difference from a plain mean is second
order in the replicate spread: under 0.1% at the ~3% spread most R0 rows show,
about 14% on a film set as inconsistent as sample 12's. It is one config key per
objective if the group prefers otherwise.

`replicate_spread` is the raw material for open issue 5 and is already in the right
space: a sd of `log T` for thickness, a sd of the value itself for the other two.
It is NaN for a single film, which is honest — one film measures no
reproducibility at all.

## Synthetic acceptance test

`tests/test_dtlz2_acceptance.py` runs DTLZ2 (3 objectives, 10 inputs, known
Pareto front) end to end through `campaign.py`. It exercises the algorithm with
no dependence on whether the experimental measurements are right.

```bash
pytest tests/test_dtlz2_acceptance.py -m "not slow"   # 10 tests, ~12 s
pytest tests/test_dtlz2_acceptance.py -m slow         # BO vs random, ~33 s
```

Measured on the negated DTLZ2 (max_hv = 0.807):

| | R0 (15) | +R1 (5) | +R2 (3) |
|---|---:|---:|---:|
| hypervolume | 0.507 | 0.555 | 0.612 |

Batch spacing: R1 min pairwise 0.735, R2 0.859, against a configured floor of
0.15 -- local penalization is separating candidates, not merely not failing.

**Cumulative hypervolume rises monotonically by construction**, so that alone is
not evidence of optimisation -- it would hold for random sampling too. The
informative result is the baseline comparison at equal budget (8 extra points
from the same 15-point start):

| | mean HV gain |
|---|---:|
| Bayesian optimisation | **+0.075** |
| random on-grid search | +0.056 |

A ratio of **1.35x**, and BO wins on **5 of 8 seeds** -- on the mean, not every
seed. With 8 added points in 10 dimensions that is the honest expectation, so the
test asserts the mean and not a per-seed win.

Two conventions that fail *silently* if got wrong, both now covered:

* DTLZ2 minimises by default; `negate=True` is mandatory or the test measures the
  opposite of optimisation.
* BoTorch's `Hypervolume` assumes maximisation and **silently drops points that
  do not dominate the reference** -- no warning, no exception, just a smaller
  number or 0.0. The helper asserts at least one point dominates before
  trusting the result.

## Open issues -- read before trusting a batch

1. **An unexplained 0.089 discrepancy on optoelectronic.** Two implementations of
   the same pipeline on the same 15 rows give LOO R2 +0.355 (two-stage) and
   +0.267 (mean module). Ruled out: the mean feature (verified raw `anneal_temp`,
   not logged) and sampling noise (no resampling involved). Also ruled out: the
   residual-vs-target standardization scale, whose direction contradicts the
   observed asymmetry. Untested suspects: MLL optimiser seeding, and the
   training-target interaction with the fitted outputscale. Both numbers are far
   better than plain (-0.342), so the direction is not in doubt -- but the gap
   should be closed before optoelectronic candidates are acted on.

2. **Done, 2026-07-30 — kept here because the audit is the evidence for how the
   objectives are now computed.** Three of the workbook's derived columns are
   pasted literals, not formulas. Audited on all 15 rows, 2026-07-29:

   | col | quantity | kind | agrees with recomputation |
   |---|---|---|---|
   | `Z` | `Uniformity score` | formula `=L2*N2*O2` | exactly |
   | `R` | `log10(P*Q)` | formula `=LOG(P2*Q2)` | 1.8e-15 |
   | `AA` | `Optoelectronic score` | **literal**, copy of R | 1.8e-15 |
   | `Y` | `Normalized thickness` | formula on **X** | — |
   | `AB` | `Thickness score` | **literal**, from the **unrounded** T mean | 4.8e-10 |
   | `X` | `Thickness (avg)` | **literal**, `ROUND(mean(T1..T4))` | 0.5 nm |

   Two things this changes. First, **`AB` is not a copy of `Y`**: `Y` evaluates
   the Gaussian on the rounded `X`, while `AB` was pasted from the same Gaussian
   on the unrounded T1..T4 mean. They disagree by up to **1.7e-3** already
   (sample 8: 0.651997 against 0.653702). The campaign path reads neither -- it
   trains on `X` -- so this is harmless there. `scripts/gp_diagnostic.py` does read
   `AB` (its `OBJECTIVE_COLS` are Z/AA/AB), where 1.7e-3 is immaterial to a
   variant comparison. Harmless either way today, but it is the same silent
   divergence that produced the original uniformity discrepancy, sitting in the
   file right now.

   Second, **the column the GP trains on is itself derived and rounded.** `X` is
   `mean(T1..T4)` rounded to whole nanometres (sample 4: 663.75 -> 664; sample
   12: 1154.5 -> 1155). Against `sigma = 176.8` nm a 0.5 nm error moves the
   utility by under 1e-5, so this is immaterial numerically. It is worth knowing
   that no raw measurement column feeds the model directly.

   **What was done.** `src/mobo_kit/scores.py` computes all three objectives from
   the measurement columns; `Z`, `R` and `X` became cross-checks that warn on
   disagreement, with a per-column tolerance because a live formula and a
   deliberately rounded literal do not deserve the same one. On the R0 rows the
   recomputation reproduces `Z` to 1.1e-16, `AA`/`R` to 1.8e-15, and `X` to the
   0.5 nm its rounding allows, so nothing about the campaign's numbers changed
   except that thickness is now unrounded. The formulas came from
   `git show pre-cleanup-2026-07-29:src/mobo_kit/d2d_scores.py` with the polarity
   inverted.

   **`Y` and `AB` are deliberately not cross-checked.** They live in utility
   space, and a check would have to duplicate the Gaussian that `objectives.py`
   owns. Nothing reads them now, so there is no dependency to protect — the
   1.7e-3 divergence above is recorded rather than monitored. If a future reader
   ever needs them, check them through `ObjectiveTransform.transform` rather than
   re-implementing the transform in `scores.py`.

3. **openpyxl discards cached formula values on save.** Verified: Z2:Z4 read
   `[0.657, 0.587, 0.561]` before a save that only added an empty sheet, and
   `[None, None, None]` after. This is why `workbook_io` writes candidates to a
   sibling file and never opens the source for writing. Do not "simplify" that
   by adding sheets to `Summary Table.xlsx`.

4. **Done 2026-07-30 — the review artifact exists; the human review itself is
   still owed.** `batch_review.py` writes a `Review` sheet into the candidate
   workbook and echoes it into the launcher pane: proposed conditions in physical
   units, predicted utility and sd per objective through the acquisition's own
   posterior-sample path, the prediction decoded into the measurement's units
   (median plus a 68% interval, multiplicative for the log-link thickness),
   normalised distance to the nearest observed point, and which coordinates sit at
   a range edge rather than only how many. Findings from Sheet1 travel with it, so
   the sheet can be forwarded on its own.

   **What the first artifact said about the R0-trained batch**, on the two flags
   raised earlier:

   * `speed_1 = 1000` — the declared probe moves each candidate to the corner and
     compares. Thickness utility falls from 0.786 to 0.223 while the sd ratio is
     **1.02**: the region is not being skipped as unexplored, it is being skipped
     as known and bad. `speed_1` is a feature of the thickness mean function, so
     that confidence is a fitted global trend extrapolating to its range edge, not
     a local average of samples 1 and 12 — and the two points anchoring that edge
     disagree, one of them (sample 12) holding `ROUND(mean(1600, 709))`. So the
     corner is a measurement question, as suspected, but by a different route than
     "the contradiction was averaged into confidence".
   * `anneal_temp` at 100–105 in all five conditions is a declared standing note:
     a monotone linear mean puts the optimum at a range edge by construction. The
     open question is chemical, and if a floor exists it belongs in `constraints:`.

   Probes and notes are declared in `configs/…yaml` under `review:`, not hardcoded.

5. **Done 2026-07-30 — `metrics.compute_ref_pareto_hv` required an explicit
   reference.** The `ref_point_np=None` path used `mins - 1e-8`, essentially the
   nadir itself: measured HV 6e-8 against 1.448 from `infer_reference_point` on
   the same data, and re-derived per call so hypervolumes were not comparable
   across iterations. Passing no reference now raises and names
   `reference_point_utility`; a reference that nothing dominates also raises,
   instead of returning the 0.0 that BoTorch's silent point-dropping produces.

   The precise condition, pinned in `tests/test_metrics.py`: `mins - 1e-8` is
   harmless while some *dominated* point sets the per-objective minima, and
   collapses once the Pareto set itself sets them — each point best in one
   objective and worst in another, which is what a genuine trade-off front is.
   Plotting code may now simply pass `config["reference_point_utility"]`.

6. **New, found 2026-07-30: the signal-collapse guard cannot tell a collapsed GP
   from a mean function that works.** `_assert_signal_not_collapsed` compares
   `gp.posterior(X).variance` against the fitted noise. A mean module does not
   enter the variance, so when a structured mean explains most of the data the
   residual GP's latent sd goes to ~0 and the guard raises `ModelFitError` —
   with a message asserting "its posterior mean is effectively constant", which is
   verifiably false in that case, because `posterior().mean` carries the fitted
   trend.

   Two situations share one numeric signature:

   * **True collapse** (the documented one): zero-mean GP, outputscale → 0,
     posterior mean genuinely flat, acquisition meaningless. Must fail.
   * **The mean function did its job**: residual variance ~0, posterior mean
     tracks the trend, candidate ranking still works — only the UCB exploration
     term has degenerated. Currently also fails, which blocks the round.

   Not reachable on the current R0 data, and reproducible on synthetic data whose
   thickness follows `log T ~ log(speed_1) + log(precur_conc)` closely (it is why
   `tests/test_batch_review.py` builds data with deliberate residual structure).
   **The risk rises with better data**, so this matters for the intake path: if the
   group returns cleaner thickness measurements, the trend may explain more and the
   launcher would refuse to propose a round.

   Suggested fix, not applied — the guard is deliberate and its rationale is
   measured, so this is a decision rather than a cleanup: test what the message
   claims. Raise only when the latent sd is negligible **and** the posterior mean
   is near-constant across the evaluated points; when the mean varies, record a
   loud `ModelFitWarning` instead, since the exploration term really has
   degenerated even though the model is usable.

7. **Not started:** replicate-variance pooling into `train_Yvar` (Phase 4) — and
   `read_candidate_results().replicate_spread` now hands it the numbers. The
   tkinter launcher landed 2026-07-30 (`launcher.py`, plus the two double-click
   scripts; see the README). The legacy debug ceremony is already gone:
   `production_gate.py` and 22 other Step 1/2A/2B/2C modules were removed in
   `33f101f`, and `test_validity_report_carries_no_approval_flags` holds the
   approval tiers out.

## Reproducing the analysis

```bash
python scripts/gp_diagnostic.py --variants legacy_matern_no_prior dim_scaled_prior
python scripts/validate_structured_means.py
python scripts/thickness_objective_check.py
```

These need the ignored private workbook at `local_inputs/Summary Table.xlsx`.
