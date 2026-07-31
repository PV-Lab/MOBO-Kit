# Round simulation: Annie's branch → `colin`

For Annie Xu. This is a review of `examples/round_simulations.py` on
`annie/ax_plots_simulation` against the current `colin` branch, and a record of
what `scripts/plot_round_simulation.py` changed and why.

Your branch forks at `33f101f`. Twelve commits landed on `colin` after that, and
most of this list is API drift rather than anything you got wrong. Two items go
the other way: one thing you fixed is still broken on `colin`, and one thing the
brief asked me to "correct" was already correct in your code.

## 1. You found a real bug, and `colin` still has it

`campaign.run_r1_ucb` passes `observed_Y_raw` — thickness in **nanometres** —
into `ObjectiveTransform.transform`, which applies `exp()` to log-link
objectives. `exp(360…1303)` overflows the 650 nm Gaussian to exactly `0.0`. It is
finite, so the non-finite guard never fires and nothing raises.

Measured on the real workbook, R0:

| observed baseline | hypervolume | Pareto size |
|---|---:|---:|
| as `run_r1_ucb` encodes it | 0.004659 | 2 |
| link decoded once (your `_physical_to_model_output`) | 0.436442 | 5 |

Every candidate's HVI is scored against a baseline whose thickness axis is pinned
at zero. Your `_run_r1_corrected` is the fix.

**Update, later the same day: `colin` no longer has it.** The group decided the
fix was its own change, and commit `4b76670` promotes the concept in your
`_physical_to_model_output` to a public contract —
`ObjectiveTransform.encode_measurements`, with `transform_measurements` as the
one-call safe route — and has `run_r1_ucb` encode before it proposes. The
acquisition modules are untouched: `ucb_hvi.py` stays byte-identical, because the
defect was in `campaign.py` orchestration.

The R1 batch built on the mis-encoded baseline was **withdrawn and reissued**
(`R1_BATCH_WITHDRAWAL.md`): four of five conditions survived, one was replaced,
and the batch's minimum spacing fell 0.9209 → 0.6337. No films had been made.
`scripts/plot_round_simulation.py` now simply calls the public `run_r1_ucb` —
verified to reproduce its own private version hash-for-hash — and keeps the
contrast in the manifest as a standing tripwire.

R2 is unaffected: `run_r2_qlognehvi` passes `train_X_norm`, and qLogNEHVI derives
its baseline through the model in model space.

## 2. Your "physical mean" label was right; the change is substantive

The brief I was given said your colorbar said "physical mean" without the
`exp(μ + v/2)` correction. It does not — `_simulate_oracle` and
`_objective_surface_values` both apply `torch.exp(mu + 0.5 * variance)`, which is
the lognormal mean, so your label was accurate for what you computed.

The new script still switches to the **median** `exp(μ)`, for a different reason
than the one in the brief. An oracle built on `exp(μ + v/2)` has a value that
depends on the posterior *variance*, which is largest exactly where the 15 real
films are sparse. The simulated ground truth would then bulge in the regions the
optimiser is about to explore, so the landscape would encode where R0 happened to
look rather than what the model believes. `exp(μ)` depends on the mean surface
alone. Both are labelled "posterior median" everywhere, and a test
(`test_the_oracle_reports_the_median_not_the_lognormal_mean`) pins it so nobody
switches it back without reading the reason.

Same for the slice-fixing convention: the brief said to standardise on the median
because a PDF snippet said "average". Your `_objective_surface_values` already
used `np.median(...)` with a grid snap. No change — it is kept, snap included.

## 3. API drift since `33f101f`

| your code | current API | why |
|---|---|---|
| `pd.read_excel(...)` + `campaign.model_source_columns(config)` | `read_campaign_workbook(path, config)` → `contents.model_values` | three of the workbook's derived score cells are **pasted literals**, not formulas, so they do not update when the measurements behind them change. `scores.py` now recomputes all three objectives from the raw measurement columns and demotes the stored cells to cross-checks. `CAMPAIGN_STATUS.md` issue 2 has the audit. |
| — | `assert contents.errors == ()` before fitting | fail closed. `contents.findings` also carries cross-check mismatches, operator-excluded readings, and films whose thickness readings disagree. |
| thickness from column `X` = `ROUND(mean(T1..T4))` | unrounded mean of whichever of `T1..T4` were measured | 7 of 15 rows change, by ≤0.50 nm. That alone moves LOO R² by 0.067, so the number matters even though the utility barely moves. |
| `getattr(campaign, "_fit_models")`, 5 positional args, returns a model | `fit_campaign_models(config, X_phys, Y_raw, seed=...)` → **`(model, warnings)`** | `_fit_models` is private, now takes `Yvar_model`, and returns a 3-tuple. The public function returns the fit guard's own findings, which must be read: a fit can succeed and still deserve distrust. |
| — | abort if the oracle fit warns | a grid built on a collapsed oracle must not render silently. The new script hard-stops rather than bannering, because every downstream number would be built on that fit. |
| `config.get("reference_point_utility")` read directly | `metrics.compute_ref_pareto_hv(Y, ref)` | it now **raises** on `ref_point_np=None` instead of using `mins - 1e-8` (measured 6e-8 against 1.448 on the same data), and raises when nothing dominates the reference instead of returning BoTorch's silent `0.0`. |
| `run_r2_qnehvi` + `src/mobo_kit/qnehvi_batch.py` | qLogNEHVI only | scope decision from the brief. qLogNEHVI is the numerically stable formulation of the same acquisition; your `campaign.py` edit is not carried over, so `colin`'s `campaign.py` stays untouched. |

## 4. Structural change: where the 2-D lives

Your script builds a **pair-slice config** per input pair — the other eight inputs
collapse to single-value grids at the experimental midpoint, the pair is
restricted to the workbook min/max — and runs a whole LHS→R1→R2 campaign inside
that 2-D slice. So each figure is its own optimisation.

The new script runs the campaign **once per parameter cell in full 10-D**, and
the 45 input pairs are *views* of that one result. Three consequences worth
knowing:

- R0 is the **real 15 recipes**, oracle-scored, not a fresh LHS. The loop then
  lives on one consistent landscape instead of mixing a measured R0 with a
  simulated R1/R2.
- the batch-identity question ("did radius 0.05 and radius 0.45 propose the same
  five conditions?") is answerable, because there is one batch per cell rather
  than 45 unrelated ones.
- it is ~45× cheaper, which is what makes 13 parameter cells affordable.

Your layout is preserved: `{pair}/qlognehvi/radius_*__beta_*/` for the surfaces,
your `_slug_number` rule (`0.25` → `0p25`), and your round legend verbatim —
"R0 LHS (GP_exp scored)", "R1 simulated", "R2 simulated". Per-condition artifacts
that have no pair (boxplots, the HV line, `all_rounds.csv`) go under
`by_condition/qlognehvi/radius_*__beta_*/`.

## 5. Smaller things

- **Outputs are gitignored.** Everything lands under `local_outputs/`, not
  `results/`. Your committed `results/` PNGs are fine on your fork and would be
  bloat on `colin`; they are not merged.
- **Boxplot fliers are off.** Every raw point is already overlaid, so matplotlib's
  flier markers drew a second, differently-styled copy of the same observation.
  The overlay itself is your convention and is kept — a box over three numbers
  reports little more than those numbers.
- **Palette** is `scripts/plot_dtlz2_report.py`'s, so every figure this project
  ships reads as one set. R0's categorical blue is the same hue the magnitude ramp
  is built from, so markers carry a white stroke around a dark edge; a single
  white edge disappears at the dark end of the ramp.
- **Footer.** Two fixed caveat lines on every figure. The oracle caveat is on all
  of them; the other line is whichever is true of that figure — the slice caveat
  on slice figures, a small-n caveat on the round summaries. The brief asked for
  one identical footer everywhere, but printing a slice caveat on a boxplot puts a
  false statement where a reader looks for true ones.
- **`min_batch_distance` is pinned at 0.15** in every cell, so "spacing" means one
  thing across the sweep. Only `radius` and `beta` move.

## 6. What did not need changing

`_safe_filename`, `_slug_number`, the directory convention, the round legend, the
overlaid boxplot points, the median-with-grid-snap slice fixing, and the decision
to fit the oracle once and freeze it. All carried over.
