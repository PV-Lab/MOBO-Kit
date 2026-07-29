# Campaign status and how to use it

Snapshot for collaborators. The full loop runs: R0 LHS -> R1 UCB-HVI (5) ->
R2 qLogNEHVI (3), three replicate films per condition, 23 distinct conditions.

## Running a round

```python
from mobo_kit.campaign import load_campaign_config, run_r0_lhs, run_r1_ucb, run_r2_qlognehvi

config = load_campaign_config("configs/FA0.9CS0.1PbI3_260407_Config.yaml")

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

**Not the three final scores.** Column order comes from
`model_source_columns(config)`, currently:

```
("Uniformity score", "Optoelectronic score", "Thickness (avg)")
```

Thickness is in **nanometres**, because the GP trains on the raw measurement and
the 650 nm Gaussian is applied to the posterior. See `GP_MODEL_DECISION.md` for
why. Anything that collects data for the next round must collect nm.

## For the plotting work

**Contour slice through the GP.** Fit with the same path a round uses, then
evaluate on a 2-D grid with the other eight inputs held fixed:

```python
from mobo_kit.campaign import _fit_models, build_objective_transform, _normalise
from mobo_kit.design import build_design_from_config

design = build_design_from_config(config)
model = _fit_models(config, X_phys, _normalise(design, X_phys), Y_model, seed=73)

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
  observed range. `_normalise(design, X_phys)` is the conversion.

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

2. **`Optoelectronic score` (column AA) is a pasted literal, not a formula.**
   R2 holds `=LOG(P2*Q2)`; AA holds a frozen copy of its value. Editing P or Q
   will not update AA. This is the same failure that produced the original
   uniformity discrepancy. The fix is to compute all three objectives in Python
   from the literal measurement columns (L/N/O, P/Q, X) and demote the formula
   columns to a cross-check that warns on disagreement. Not yet done. Column AB
   deserves the same audit.

3. **openpyxl discards cached formula values on save.** Verified: Z2:Z4 read
   `[0.657, 0.587, 0.561]` before a save that only added an empty sheet, and
   `[None, None, None]` after. This is why `workbook_io` writes candidates to a
   sibling file and never opens the source for writing. Do not "simplify" that
   by adding sheets to `Summary Table.xlsx`.

4. **No batch has been reviewed by a human.** Boundary counts and pairwise
   distances look healthy ([3,4,2,2,2], min 0.921), but nobody has inspected the
   five proposed conditions in physical units. Fifteen films is a real cost.
   One specific thing to look for: whether anything lands near
   `speed_1 = 1000`, a region with two contradictory observations in it.

5. **Not started:** the tkinter launcher, replicate-variance pooling into
   `train_Yvar` (Phase 4), and removal of the legacy Step 2C debug ceremony.

## Reproducing the analysis

```bash
python scripts/gp_diagnostic.py --variants legacy_matern_no_prior dim_scaled_prior
python scripts/validate_structured_means.py
python scripts/thickness_objective_check.py
```

These need the ignored private workbook at `local_inputs/Summary Table.xlsx`.
