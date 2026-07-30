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
| `diagnostics` | method, seed, pool size, objective contract version, validity report, fit warnings |

Two warning keys, deliberately separate. `diagnostics["model_fit_warnings"]` holds
only the fit guard's own findings — the ones a human reviewing a batch must read,
and the ones the launcher and the `Review` sheet surface.
`diagnostics["fit_warnings_raw"]` holds everything the fits raised, including the
~18 numpy-2.0 deprecation notices per fit that this stack emits. Nothing surfaces
the raw list; it is there for debugging a strange fit later, because a BoTorch or
scipy convergence warning that the filter dropped is exactly what would be wanted
then.

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
model, fit_warnings = fit_campaign_models(config, X_phys, Y_model, seed=73)
assert not fit_warnings          # a fit can succeed and still deserve distrust

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

1. **The 0.089 discrepancy on optoelectronic — narrowed 2026-07-30, not closed.**
   Two implementations of the same pipeline on the same 15 rows give LOO R2 +0.355
   (two-stage) and +0.267 (mean module). Reproduced exactly: **+0.0881**.

   **MLL optimiser seeding is ruled out.** Both pipelines give bit-identical LOO R2
   across seeds 7, 73, 137 and 2024 — 0.3551 and 0.2670 every time, zero variation.
   That suspect is closed.

   **The standardization-scale suspect is back, and quantitatively consistent.** It
   was previously recorded as ruled out "because the direction contradicts the
   observed asymmetry"; the measured direction does not contradict it. The two
   pipelines hand `Standardize` different things — two-stage standardizes the
   *residual*, the mean module standardizes the *target* and then subtracts a
   standardized trend — so the deviation the covariance must explain has sd 1.0 in
   one and `sd(residual)/sd(target) = 0.762` in the other. The fitted outputscales
   match that prediction to 4%:

   | | median outputscale | median noise (standardized) |
   |---|---:|---:|
   | two-stage | 0.8365 | 0.006516 |
   | mean module | 0.4681 | 0.006443 |
   | predicted for the mean module, `0.8365 × 0.762²` | 0.4859 | — |

   **The attempt to confirm it failed, and the test was the problem, not the
   hypothesis.** Inflating the residual to the target's sd before fitting moved LOO
   R2 by +0.0002 — because `Standardize` divides by whatever sd it is given, so
   scaling its input is a no-op. That experiment was vacuous by construction and
   proves nothing either way. Recorded so nobody re-runs it.

   **The specific next test**, for whoever picks this up: the two pipelines cannot
   be separated while both re-standardize, so disable `Standardize` in both (or
   standardize both by the same fixed constant) and see whether the gap survives.
   If it vanishes, the cause is that the outputscale and noise priors are defined
   on standardized units and the two pipelines standardize different quantities.
   That is a ~20-line experiment against `_build_single_task_gp`.

   Both numbers remain far better than plain (-0.342), so the direction is not in
   doubt and the mean module stays either way. The gap should be closed before
   optoelectronic candidates are acted on.

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

6. **Done 2026-07-30 — the signal-collapse guard now distinguishes a collapsed GP
   from a mean function that works.** It used to compare
   `gp.posterior(X).variance` against the fitted noise and stop there. A mean
   module does not enter the variance, so when a structured mean explains most of
   the data the residual GP's latent sd goes to ~0 and the guard raised
   `ModelFitError` — asserting "its posterior mean is effectively constant", which
   is verifiably false in that case, because `posterior().mean` carries the trend.

   Two situations share one numeric signature and now get different answers:

   * **True collapse**: zero-mean GP, outputscale → 0, posterior mean genuinely
     flat, nothing can be ranked. Still `ModelFitError`.
   * **The mean function did its job**: residual variance ~0, posterior mean
     tracks the trend, ranking still works. Now a loud warning and the round
     proceeds. Refusing would dead-end the campaign at the moment the physics model
     started working, with no remedy — better data cannot be collected without
     first proposing conditions. The review artifact is the designed gate.

   The warning is not a formality, and says so: UCB's exploration term reads the
   latent posterior that just collapsed, and the mean module's coefficients are
   frozen buffers with no uncertainty of their own, so the narrow intervals such a
   model reports are **understated rather than earned**. It appears above the
   numbers in both the launcher pane and the `Review` sheet, and in
   `RoundResult.diagnostics["model_fit_warnings"]`.

   Two calibration notes worth keeping:

   * "Near-constant" is measured against the **observed spread of that
     objective**, not against the fitted noise sd. Noise-relative was the first
     attempt and is wrong: the noise is inflated precisely in the degenerate case,
     so the test co-varies with what it is trying to detect. Measured instance — a
     linear mean on `anneal_temp` against a forced noise of 0.9 scored 0.38 on the
     noise yardstick and would have been called constant while it was tracking the
     data. Floor is 5% of the observed spread.
   * Only the guard's own warnings reach a human. `record.warnings` also collects
     every Python warning raised during fitting — about 18 numpy-2.0 deprecation
     notices per fit on this stack — and putting those in front of someone
     reviewing a batch is how people learn to ignore warnings.

   Whether a given dataset trips the collapse is knife-edge: measured across
   residual magnitudes from 0 to 0.3 it fires at 0, 1e-4, 0.01 and 0.03 but not at
   0.001 or 0.1, because it depends where the MLL optimiser lands. The guard's
   decision is therefore tested directly, and the propagation tests force the
   condition rather than hoping data produces it. No fit on the current R0 data
   warns, so nothing about the live campaign changed.

7. **Phase 4 is wired and waiting for data (2026-07-30).** `replicate_variance.py`
   pools between-film variance from the replicate scatter and hands it to the model
   as `train_Yvar`; `run_r1_ucb` / `run_r2_qlognehvi` / `fit_campaign_models` take
   `observed_Yvar`, and the launcher builds it automatically once the config asks.
   Enabling it when the triplicates land is one key —
   `model.observation_noise: replicate_pooled` — which is the point of wiring it
   before the data exists. Tested against synthetic replicates.

   Four things worth knowing before touching it:

   * **The variance handed over is of the MEAN**, `pooled / n_films`, because the
     observation is an average of n films. Passing the single-film variance
     understates it threefold on a triplicate and nothing errors.
   * **Between-film and within-film are different quantities.** Between-film is
     what `train_Yvar` needs. The within-film 0.0593 on `log T` (24 dof) contains
     no run-to-run variation at all, so it is a **floor**: if the pooled
     between-film variance ever lands below it, films would be more reproducible
     than points on one film, and `sanity_floor_findings` says so.
   * **BoTorch silently ignores `train_Yvar` if a `likelihood` is also passed.**
     Verified on 0.15.1: the likelihood wins, stays single-element, and the
     replicate information is dropped with no error. `_build_single_task_gp` passes
     one or the other, never both.
   * **`Standardize` rescales `train_Yvar` along with the targets**, so it must
     arrive in the target's own units — and in the model's space, which for
     thickness is `log T`, not nanometres. That is why aggregation and variance
     pooling are required to share one space.

   Zero pooled variance is refused rather than passed on: replicate films that
   agree to the last digit are a transcription, not a measurement, and a zero
   `train_Yvar` tells the model the observation is exact.

8. **Not started:** the legacy leftovers below. The
   tkinter launcher landed 2026-07-30 (`launcher.py`, plus the two double-click
   scripts; see the README). The legacy debug ceremony is already gone:
   `production_gate.py` and 22 other Step 1/2A/2B/2C modules were removed in
   `33f101f`, and `test_validity_report_carries_no_approval_flags` holds the
   approval tiers out.

## Are beta = 4.0 and radius = 0.25 defensible?

`scripts/dtlz2_parameter_sweep.py`, 8 seeds per cell, `min_batch_distance` fixed at
0.15. Metric is mean hypervolume gain over the R0 start for the 8 points R1 and R2
add, against a random on-grid baseline at the same budget (+0.0453 in every cell,
since it does not depend on either knob).

| beta | radius | mean gain | per-seed sd | min spacing | edge coords / 80 |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.15 | +0.0816 | 0.0508 | 0.719 | 16.4 |
| 2 | 0.25 | +0.0781 | 0.0493 | 0.810 | 16.5 |
| 2 | 0.35 | +0.0801 | 0.0481 | 0.955 | 16.9 |
| 4 | 0.15 | +0.0776 | 0.0440 | 0.719 | 16.0 |
| **4** | **0.25** | **+0.0780** | **0.0428** | **0.891** | **16.8** |
| 4 | 0.35 | +0.0961 | 0.0735 | 0.982 | 17.0 |
| 8 | 0.15 | +0.0868 | 0.0523 | 0.871 | 16.9 |
| 8 | 0.25 | +0.0868 | 0.0523 | 0.871 | 16.9 |
| 8 | 0.35 | +0.0839 | 0.0458 | 0.953 | 17.2 |

**No change.** The pre-committed rule required a challenger to beat +0.0780 by more
than the per-seed sd of 0.0428 — that is, to exceed +0.1208 — without reducing
spacing; seven cells have a higher mean and none comes close, the whole grid
spanning +0.0776 to +0.0961 against sds of 0.043 to 0.074. BO beats the random
baseline on the mean in 9 of 9 cells, so the sweep is measuring optimisation rather
than noise, and the edge-coordinate count is flat at 16–17 of 80 across every cell,
which says neither knob is what drives batches onto range edges (on the live
campaign that was the monotone `anneal_temp` mean function).

**One limit worth stating**: `radius` is not binding on this problem. Achieved
batch spacings are 0.72–0.98, far above every radius tested, so local penalization
rarely has two candidates close enough to penalise — visible in `beta=8` giving
identical results at radius 0.15 and 0.25. This sweep therefore validates `beta`
properly and says little about `radius`; a problem with a tighter optimum would be
needed for that.

## When new data arrives

One command:

```bash
python scripts/intake_new_data.py --workbook "local_inputs/Summary Table.xlsx"
```

The group has always called the current numbers test data, so a replacement was
expected. When it lands, the question is not whether the code runs — the tests
answer that — but whether the model commitments this campaign made still earn
their place on the new rows. Several were justified by measurements on 15 specific
rows and do not transfer.

It prints, per objective: the read audit and its findings; whether the declared
`mean_function` still beats the leave-one-out null by more than the resolution
floor, naming the exact config block to delete if not; the fit guard's status,
including the case where the mean function explains so much that the residual GP
collapses; whether the fixed anchors still span the data; and whether the
campaign-fixed scaling guard passes.

**Both floors are recomputed at the new N rather than reused.** The null is
`1 - (N/(N-1))²` — −0.148 at 15, −0.105 at 21, −0.069 at 31. The ±0.236 resolution
figure was a bootstrap at N=15 and is rescaled by `sqrt(15/N)`, labelled in the
output as an estimate: re-run the bootstrap if a decision turns on the third
decimal.

On the current 15 rows it reports: uniformity does not beat the null (−0.681),
optoelectronic keeps its mean function (−0.342 → +0.267, swing +0.609), thickness
keeps its mean function (+0.116 → +0.381, swing +0.265). The guard is clean for
both.

## Reproducing the analysis

```bash
python scripts/gp_diagnostic.py --variants legacy_matern_no_prior dim_scaled_prior
python scripts/validate_structured_means.py
python scripts/thickness_objective_check.py
python scripts/dtlz2_parameter_sweep.py            # beta x radius, needs no data
```

All but the sweep need the ignored private workbook at
`local_inputs/Summary Table.xlsx`.
