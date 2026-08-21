# Campaign status and how to use it

Snapshot for collaborators. The full loop runs: R0 LHS -> R1 UCB-HVI (5) ->
R2 qLogNEHVI (3), three replicate films per condition, 23 distinct conditions.

## The live campaign, from 2026-08-17 — READ THIS FIRST

**A second dataset arrived and it runs a different objective contract.** Two of
the three objectives are computed differently, the workbook's columns moved, two
grids changed, and this project's first real constraints are active. Everything
below this section describes the FIRST campaign unless it says otherwise, and its
numbers are about that contract.

| | first — algorithm testing | second — **the live campaign** |
|---|---|---|
| config | `configs/campaign_d2d_perovskite.yaml` (**archived**) | `configs/campaign_d2d_perovskite_test.yaml` |
| contract | `d2d-objectives-v2-nm-thickness` | `d2d-objectives-v3-test` |
| workbook | `local_inputs/Summary Table.xlsx` | `local_inputs/Summary Table Test.xlsx` |
| uniformity | `Coverage * (1-Uniformity) * Phase purity` | `mean(Coverage, 1-clamp(Uniformity), Phase purity)` |
| optoelectronic | `log10(Voc * Photoconductance)` | `mean(min(Voc,1.4)/1.4, Normalized photoconductance)` |
| thickness | mean of `T1..T4`, nm | unchanged |
| constraints | none, deliberately | three, active |

The old config is archived rather than deleted, and stays complete and loadable:
every number in `GP_MODEL_DECISION.md` is about that contract. Archived means "do
not run new rounds against it".

**Why a new file and not an edit.** Utility space is what hypervolume is measured
in. An objective that keeps its name while changing its construction makes every
cross-campaign number incomparable while every plot still renders — which is the
failure mode the `contract_version` key exists to prevent.

**All three recipes reproduce the stored score columns**, worst disagreement
2.3e-13 across all 15 rows. Per the group, for this workbook the stored scores are
authoritative and the recompute is the cross-check, so a disagreement is a warning
finding rather than a block.

### What the second dataset supports

`scripts/intake_new_data.py`, 2026-08-17, exact leave-one-out, null −0.1480 at
N=15, resolution floor ±0.236:

| objective | plain GP | with mean function | verdict |
|---|---:|---:|---|
| uniformity | **−0.6447** | — | below the null, **exploration only** |
| optoelectronic | **−0.5842** | −0.6977 | below the null, **exploration only**, mean function **deleted** |
| thickness | **+0.5227** | **+0.6630** | **learnable**; the swing is inside the floor |

**Two of the three axes carry no signal.** A batch is therefore chosen on one
informative axis and two uninformative ones. That is a legitimate exploration
round, but it is not a three-objective optimisation, and the review must say so
rather than let the predicted numbers imply otherwise.

**The optoelectronic mean function was deleted, and that is the designed
outcome.** The first campaign's linear `anneal_temp` trend was worth −0.342 →
+0.355 on its own score; here it makes the fit *worse*, −0.5842 → −0.6977. The
target was redefined underneath it, so the old evidence was never about this
quantity. Do not reinstate it from the archived config without a fresh verdict.
Issue 10 is the prime suspect for why the objective is unlearnable at all.

**Thickness keeps its mean function, decided by the rank permutation.** The plain
GP now reaches +0.5227 where the first campaign's managed +0.116, so the trend has
much less left to explain and the +0.1403 swing is inside the ±0.236 floor —
*inconclusive on R²*, which is a statement that R² cannot resolve it at N=15
rather than a verdict. `scripts/permutation_rank_test.py` adjudicated it on
2026-08-18:

| | value |
|---|---:|
| observed rank ρ | **+0.7250** |
| null mean (sd) | −0.1917 (0.2937) |
| exceedances | **4 of 1800** |
| p | **0.0028**, 95% CI [0.0003, 0.0052] |

Stronger than the first campaign's p = 0.0350 on its own films. **Rank is the
right statistic because rank is what the acquisition consumes** — it never sees
R². **Do not quote +0.1403 as evidence**; the permutation is what carries the
weight, and the swing is merely consistent with it.

That two-part rule is now what the intake prints: (i) the structured fit must beat
the null by more than the floor; (ii) when structured-versus-plain lands inside the
floor, the permutation decides.

### Are beta = 36 and radius = 0.35 defensible?

**The live campaign runs beta = 36 and radius = 0.35.** They were determined by a
sweep over two instruments on the campaign's own data — per-round utility **box
plots**, and **heat maps**, which are 2-D slices through the higher-dimensional
Gaussian-process model — across **beta from 9 to 49** (9, 25, 36, 49) and **radius
from 0.05 to 0.45** (nine values, step 0.05), three starting designs per cell at
production settings. **Note that local penalization is inert at the current beta**;
the measured consequences are below. Outputs stay local (`local_outputs/`): they
are how the group picks a setting, not a result about the chemistry, and they are
not part of what this repository publishes.

**They are a declared policy choice, not a measured optimum, and the distinction
matters.** That sweep **could not rank cells**: the whole spread across betas was
0.0065 against a trial-to-trial sd of 0.010–0.027, and the best cell was a
different (β, r) in every trial. It also scored candidates against a *noiseless
GP oracle of the same model class the optimiser fits*, so the landscape held no
surprises and exploration had unusually little to earn — it **systematically
undervalues large β**, which is the very thing this cell buys.

The rationale is a posture, not a score: β = 36 means κ = √36 = 6, heavy
exploration, which is the right stance when **two of three objectives carry no
learnable signal** and the third is the only one worth exploiting.

**Two consequences are on record, both measured on the live R1 proposal.**

* **Local penalization is inert at this β.** Achieved minimum batch spacing is
  **1.091**, three times the 0.35 radius, so the knob has nothing to act on. The
  sweep predicted this: radius binds *less* as β rises, and at β = 49 the nine
  radii produced only 3–4 distinct batches.
* **The batch runs to the edges.** Range-edge coordinates per condition are
  **[4, 7, 4, 3, 3]** — 21 of 50 — against 11–15 of 80 on the first campaign's
  arm at β = 4. High exploration plus a monotone thickness trend puts candidates
  at bounds. That is expected, not a defect, but it is what a reviewer should
  check before fabricating.

**Two triggers to revisit.** First, when the photoconductance normalisation is
fixed (issue 10) and optoelectronic may become learnable — the posture was chosen
for two dead axes and would no longer be justified by the same argument. Second,
when R1 measurements land and the oracle can be replaced by real film-to-film
noise, at which point the sweep's central caveat stops applying and a genuine
ranking becomes possible.

### The simulation at the ratified cell

`scripts/plot_round_simulation.py --cell 0.35,36` runs one campaign against the
frozen oracle at exactly the decided knobs; `scripts/plot_boxplot_sweep.py
--betas 36 --radii 0.35` runs the same cell across the three starting designs so
the per-round boxes have a distribution behind them. Both default to the v3
config. Outputs: `local_outputs/round_sim_v3_cell` and
`local_outputs/boxplot_v3_cell`.

Measured on the new data, seed 73:

| | R0 | +R1 | +R2 |
|---|---:|---:|---:|
| hypervolume | 0.7929 | 0.7929 | 0.8053 |

**R1 adds no hypervolume at all on this oracle, and R2 adds +0.0124.** That is
what β = 36 looks like against a landscape with no surprises in it: the batch
spends its budget on exploration that a noiseless same-class oracle cannot repay.
It is the caveat above made numerical — the instrument understates the case for
the policy it is testing — and not evidence that the cell is wrong.

**The cross-instrument identity holds.** The simulated R1 batch hashes to
`60d1682aa055ca97`, the same as the live `run_r1_ucb` proposal from the measured
rows. The simulation is describing the batch the campaign would actually ship, not
a similar one.

Pre-registered expectations, checked after: the two no-signal axes climb far less
than thickness (+0.1047 and +0.0842 against +0.3967) — **HELD**; the sweep-arm
rules report **NOT APPLICABLE** rather than FAILED, because a single ratified cell
has no arm to vary and calling that a failure would put red lines under a run that
did exactly what was asked.

The dead axes' surfaces are rendered and captioned as fitted noise, never dropped.
Thickness's surface shows the declared `log T ~ log(speed_1) + log(precur_conc)`
trend, which is **consistency with what the config told the model, not a
discovery**.

### The round report — figures at propose time

Pressing **Propose next round** now also renders six figures beside the workbook,
in `<workbook stem>_reports/<round>_<UTC timestamp>/`. A second button, **Figures
from current data**, renders the four that need no batch — use it the moment a
round's measurements are entered, before deciding whether to propose at all. Same
thing headless:

```bash
python scripts/generate_round_report.py --workbook "local_inputs/Summary Table Test.xlsx" --data-only
```

**Every figure writes the CSV behind it**, plus a `manifest.json` recording the
contract version, seed, git describe, reference point, runtime and the active
notices. A PNG whose numbers cannot be re-derived is the next
plausible-finite-number bug; this project has had three. Two equalities are
asserted by tests rather than by convention: the parity numbers *are*
`intake_new_data.py`'s numbers (one shared fold loop in `mobo_kit.loocv`, not two
implementations that agree today), and figure 03's numbers *are* the Review
sheet's.

| figure | what it shows | what it cannot claim |
|---|---|---|
| `00_batch_placement` | proposed recipes over the measured cloud, normalised to the declared grid, plus batch spacing | nothing about quality — only where in recipe space the batch goes |
| `01_loo_parity` | leave-one-out prediction against measurement, per objective, with LOO R² and the null | an axis marked NO LEARNABLE SIGNAL has a model that does not beat the null; its scatter is nothing, not a weak trend |
| `02_attribution` | mean \|SHAP\| per input per objective, in utility units | explains the **model**, not the world; features in a `mean_function` were *told* to it; on a no-signal axis the bars are fitted noise |
| `03_batch_predictions` | predicted measurement and utility per condition, plus the batch's ΔHV distribution and per-candidate P(non-dominated) | predictions, not measurements |
| `04_hv_trajectory` | cumulative observed hypervolume per measured round | monotone **by construction** — random sampling rises too, so this is progress, not proof of optimisation |
| `05_objective_space` | pairwise utility panels with per-pair fronts, 3-objective front ringed, plus one fixed 3D view | the Pareto set is non-dominated among what has been **measured**, not across the design space |

**Runtime is about 15 s at N=15** on an idle machine, dominated by the 45
leave-one-out refits (9.8 s) and the attribution (a few seconds at 15 instances).
The fold loop runs single-threaded on purpose: at 14×10 the matrices are small
enough that intra-op threading costs more than it buys — 9.8 s at one thread
against 15.1 s at this box's default of 12, bit-identical either way.

*The first measurement of that recorded 51 s against 117 s and was wrong: it was
taken while sixteen permutation workers were saturating the CPU. The effect was
real but was of the load, not the thread count. A timing under contention is an
unreproduced number like any other, and this project's rule is that those get
re-measured rather than written down. Add the first render of a session to any of
these: matplotlib builds its font cache once, which cost about a minute here.*

**A report failure never costs a batch.** The worklist and the Review sheet are
written before the figures are drawn; if rendering fails, `Generated.report_error`
says so and the batch stands. Inside the report, one failed figure is recorded in
the manifest and the rest still render.

**Three notebook conventions were deliberately not ported.**

* **In-sample parity.** Asking a model about points it was fitted on measures
  memorisation; at N=15 in 10 dimensions it is close to a straight line whatever
  the model knows. Parity here is leave-one-out.
* **Ad-hoc sign flips at plot time.** Objective polarity is a config contract
  (`goal:`). Flipping a sign in a figure makes the figure disagree with the
  optimiser, and only one of them is right.
* **Auto-referenced hypervolume.** The reference point is required and
  campaign-fixed. A reference re-derived per call gave 6e-8 against 1.448 on the
  same data once already — see issue 5.

### The two grid edits

Both forced by the measured rows; everything else carries over unchanged, and all
15 rows land on the declared grid.

* **`time_2` now starts at 0** (was 10). Sample 2 is a one-step film — `speed_2`
  and `time_2` both zero — so 0 has to be on the grid or a real recipe is
  off-grid. Reaching 0 with step 5 also reaches 5, which the first campaign's grid
  excluded and no film has run, so the hole is declared as a `nonzero_minimum`
  constraint rather than filled in silence.
* **`anti_time` now steps by 1** (was 2), 9..25. Sample 1 runs `anti_time = 12`,
  which the old grid could not hold; the first campaign carried it as a declared
  off-grid exception excluded from pool bookkeeping. The axis goes from 9 values
  to 17.

An explicit value list would express `{0} ∪ {10..60}` directly and avoid the
`nonzero_minimum` workaround, but `lhs` asserts that every design grid is
uniformly spaced, so it would need changes to `design.py` and `lhs.py`. **Worth
raising with the group:** whether a 5 s second spin should ever be allowed, and
whether `anti_time` wants step 1 or an explicit list.

### The constraints

Declared in the new config, enforced by filtering the candidate pool before any
acquisition sees it, and re-checked independently by `validate_batch`. The
acquisition modules are byte-identical. `discrete_refinement` is **not**
constraint-aware and is not wired into a round; its docstring says so.

| name | rule | why |
|---|---|---|
| `second_stage_all_or_nothing` | `speed_2` and `time_2` both zero or both nonzero | a stage at 0 rpm for 30 s is a contradiction; both zero is a one-step film, which sample 2 is |
| `antisolvent_lands_while_spinning` | `anti_time < time_1 + time_2`, strictly | dropping at exactly the end is already too late |
| `second_stage_runs_at_least_10s` | `time_2` is 0 or ≥ 10 | the declared hole in the arithmetic grid, above |

All 15 measured rows satisfy all three. Observed rows are soft-checked only —
history is history, and a constraint that rejects a film the group actually ran is
far more likely to be wrong than the film is.

**Watch `constraint_pool_survival_rate`** in the round diagnostics. The sampler
draws until it has the requested pool size, so a mis-specified constraint produces
a normal-looking pool drawn from a sliver of the space, and the survival rate is
the only place that shows.

## Everything below this line is about the FIRST campaign

> **The first campaign was algorithm testing.** Its 15 rows and its
> `d2d-objectives-v2-nm-thickness` contract existed to prove the loop worked, not
> to run an experiment. The sections below are its record and its numbers are
> about *its* objectives — uniformity as a product, optoelectronic as a log10
> product — which the live campaign redefined. **Nothing here transfers unless it
> is method rather than measurement.** Where a mechanism still applies (how a
> round runs, what `Y_model` must contain, the acceptance test) it applies to
> both; where a fitted number appears, it is the first campaign's.

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

**This is now implemented.** `scripts/plot_round_simulation.py` runs the whole
loop against a frozen GP oracle and renders contour slices, per-round boxplots and
a hypervolume line, with a batch-identity manifest
(`docs/ROUND_SIM_MANIFEST.md`). Read `docs/ROUND_SIM_DELTA.md` before extending
it. The recipe below is kept because it is what any new plotting code has to get
right, and both conventions still fail silently.

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
round on shared axes (R0 blue `#2a78d6` / R1 orange `#eb6834` / R2 green
`#1baf7a` — the palette `plot_dtlz2_report.py` and `plot_round_simulation.py`
both use, so project figures read as one set), plus per-round
`min_pairwise_distance` and boundary counts from `diagnostics`. Contour slices
should show **23 distinct conditions**, not 39 films -- replicates share inputs
and would otherwise overplot.

Hypervolume is comparable across rounds only because objective scales are fixed
in config; `assert_scaling_is_campaign_fixed` enforces that. Do not re-derive
scales from observed data between rounds.

## Model state

Validated on the 15 R0 observations, exact leave-one-out, null R2 = -0.148. These
are the canonical numbers, as `scripts/intake_new_data.py` reports them — same
pipeline and same inputs the model uses:

| objective | plain GP | with structured mean | swing |
|---|---:|---:|---:|
| thickness (nm) | +0.116 | **+0.381** | +0.265 |
| optoelectronic | -0.342 | **+0.267** | +0.609 |
| uniformity | no learnable signal (permutation p = 0.82) | n/a | — |

Both swings clear the ±0.236 sampling floor. `GP_MODEL_DECISION.md` records
slightly different figures (+0.183 → +0.384 for thickness, and +0.355 for
optoelectronic); those came from an older instrument reading the workbook's rounded
`Thickness (avg)`, and both differences are accounted for — see issue 1 and the
intake section below. No conclusion depends on which set you read.

Thickness rests on its rank permutation (p = 0.0350), not on the R2 swing.
Uniformity is exploration-only by measurement, not by choice; the interface must
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

`replicate_spread` is what Phase 4 (issue 7) pools, and it is already in the right
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

## The numbered issues -- read before trusting a batch

**Issues 1-9 are the first campaign's**, kept because each one's evidence is the
reason a decision holds and because several are the sort of thing that gets
rediscovered and re-argued. **Issue 10 is the live campaign's and is open.**

Kept numbered and in place even once closed, because each one's *evidence* is the
reason a decision holds, and because several are the sort of thing that gets
rediscovered and re-argued. Status is stated at the top of each.

1. **CLOSED 2026-07-30. The 0.089 on optoelectronic is a numerical artifact, not a
   modelling difference.** The two pipelines specify *the same model*: fitting a
   zero-mean GP to `y - trend` and fitting a fixed-mean GP to `y` with mean
   `trend` have identical marginal likelihoods, because a fixed mean only shifts
   the data. So there was never a modelling question to answer — only a question
   about why two routes to one model disagreed.

   Two contributions, measured:

   | | two-stage | mean module | gap |
   |---|---:|---:|---:|
   | with `Standardize` (production) | +0.3551 | +0.2670 | **+0.0881** |
   | without `Standardize` | +0.3385 | +0.2670 | +0.0715 |

   *Standardization scale accounts for about 19%.* With the transform in place the
   two pipelines standardize different quantities — the residual in one, the target
   in the other — so the outputscale and noise priors, which are defined on
   standardized units, act on differently-scaled residuals. Removing it moves the
   gap from 0.0881 to 0.0715.

   *The remaining 81% is the MLL optimiser.* With the transform gone the likelihood
   surfaces are identical, yet the fits land in slightly different places: across
   folds the outputscale differs by up to 2.7%, the noise by 2.7%, and the median
   lengthscale by **9.6%**. At N=15 that is enough to move LOO R² by 0.07. The
   optimiser is deterministic — the earlier seed sweep found bit-identical results
   across four seeds — so this is a different starting point on one surface, not
   stochastic variation.

   **The mechanism, stated first because that is the rule.** The gap is roughly
   one-fifth a definitional difference between two legitimate conventions and
   four-fifths the optimiser landing in a different place on one identical
   objective. Both parts are named, measured and reproducible. This project's own
   rule is that a deterministic difference on the same rows must be *explained*,
   not absorbed into a floor — so the explanation comes first and the floor comes
   after it.

   **The floor, as a corollary.** Given the mechanism, 0.0881 is also inside the
   ±0.236 sampling floor, and its optimiser component is exactly the measurement
   that established the ≈0.07 numerical-reproducibility floor — see
   `GP_MODEL_DECISION.md`, "Three floors". So it was never evidence of anything.
   That is a consequence of the explanation, not a substitute for it.

   The campaign uses the mean-module convention, the one wired into `campaign.py`.
   No action. Kept below for the reasoning, because "two implementations disagree"
   is the sort of thing that gets rediscovered.

   ---

   *Original entry, narrowed 2026-07-30 before the closure above.*
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

   **The batch this described was withdrawn and reissued on 2026-07-31** — see
   `docs/R1_BATCH_WITHDRAWAL.md`. Four of its five conditions survived unchanged,
   including `R1_C01`, which is the condition the numbers below are quoted from, so
   **these figures are unchanged and were re-read from the reissued artifact rather
   than assumed.**

   **What the artifact says about the R0-trained batch**, on the two flags raised
   earlier:

   * `speed_1 = 1000` — the declared probe moves each candidate to the corner and
     compares. For `R1_C01`, thickness utility falls from 0.786 to 0.223 while the
     sd ratio is **1.02**. Across all five conditions the mean falls 0.791 → 0.299
     with a mean sd ratio of **1.10**. Either way the region is not being skipped
     as unexplored, it is being skipped as known and bad. `speed_1` is a feature of
     the thickness mean function, so that confidence is a fitted global trend
     extrapolating to its range edge, not a local average of samples 1 and 12 — and
     the two points anchoring that edge disagree, one of them (sample 12) holding
     `ROUND(mean(1600, 709))`. So the corner is a measurement question, as
     suspected, but by a different route than "the contradiction was averaged into
     confidence". The reissued batch's minimum `speed_1` is still 1500.
   * `anneal_temp` at 100–105 in all five conditions is a declared standing note:
     a monotone linear mean puts the optimum at a range edge by construction. The
     open question is chemical, and if a floor exists it belongs in `constraints:`.
     Unchanged in the reissued batch.

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
     observation is an average of n films. Passing the single-film variance would
     be three times too large on a triplicate — *overstating* uncertainty, so the
     model would trust the most carefully replicated conditions least — and nothing
     errors.
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

8. **Done — the legacy leftovers are gone.** The tkinter launcher landed
   2026-07-30 (`launcher.py`, plus the two double-click scripts; see the README).
   The legacy debug ceremony went earlier: `production_gate.py` and 22 other Step
   1/2A/2B/2C modules were removed in `33f101f`, and
   `test_validity_report_carries_no_approval_flags` holds the approval tiers out.

9. **Fixed 2026-07-31 — `run_r1_ucb` scored every candidate against a baseline
   whose thickness axis had collapsed to zero.** This is the most consequential
   defect found in this project, and the R1 batch it produced was withdrawn:
   `docs/R1_BATCH_WITHDRAWAL.md`.

   `ObjectiveTransform.transform` is a MODEL-OUTPUT decoder — it applies `exp()`
   to a log-link objective before computing utility. `run_r1_ucb` handed it
   `observed_Y_raw`, thickness in **nanometres**, so the value was exponentiated a
   second time. `exp(360…1303)` saturates the 650 nm Gaussian to exactly `0.0`.

   | | as called | correctly encoded |
   |---|---:|---:|
   | observed baseline hypervolume | **0.004659** | **0.436442** |
   | baseline Pareto set | 2 points | 5 points |

   A factor of 94, and every candidate's improvement was measured against a front
   with no thickness axis at all. On the live batch this moved one of five
   conditions and the minimum spacing from 0.9209 to 0.6337 — so it also inflated
   the spacing figure that this document used to argue `radius` was inert.

   **The fix** is `ObjectiveTransform.encode_measurements`, with
   `transform_measurements` as the one-call safe route, and `run_r1_ucb` encoding
   before it proposes (commit `4b76670`). `ucb_hvi.py` is untouched — it is one of
   the frozen acquisition modules and the defect was in `campaign.py`
   orchestration. **Annie Xu found and fixed this independently on
   `ax_plots_simulation` before we knew it existed**; the fix promotes her
   `_physical_to_model_output` to the public contract.

   **Every `ObjectiveTransform.transform` call site was audited.** `objectives.py`
   403/481/516 and `ucb_hvi.py:342` all operate on posterior samples, already in
   model space; `batch_review.py` never routes measurements through the transform
   at all. Exactly one call site was defective, `ucb_hvi.py:698`, reached only
   from `run_r1_ucb`. R2 was never affected — qLogNEHVI takes `train_X_norm` and
   derives its baseline through the model.

   **This is the third plausible-finite-number failure in this project**, after
   the hypervolume auto-reference (issue 5) and the silently swallowed
   `train_Yvar` (issue 7). All three share one shape: **a wrong answer that is
   finite, ordinary-looking, and compared against nothing.** No guard fires
   because nothing is out of range; the number is simply not the number anyone
   meant. The lesson is not "add more guards" — each of these passed every guard
   it met — it is that **a quantity no test reproduces independently is a
   quantity nobody is checking.** `run_r1_ucb` now reports
   `observed_baseline_hypervolume` and `observed_baseline_pareto_size` so the
   value is observable from outside, and the tests recompute both by a separate
   route.

   **Why 446 tests missed it.** Every objective in the synthetic acceptance test
   was affine, and for an affine objective measurement space and model space are
   the same numbers — a link-encoding mistake is invisible *by construction*.
   `test_dtlz2_acceptance.py` now also runs with a log-link objective, so every
   end-to-end pass exercises both link types the campaign uses.

10. **OPEN, second campaign. `Normalized photoconductance` does not rank like the
    photoconductance it summarises, and it is half of the optoelectronic
    objective.** Nothing in the workbook derives that column — it arrives already
    normalised, from outside — so a recipe can only take it on trust, and a
    normalisation that has come loose from its measurement is invisible: every
    value is in range, every row computes, and the objective is simply about
    something other than it says.

    Rank agreement is the check that needs no formula. Whatever the intended
    mapping is, it must preserve order. Measured over the 15 R0 rows:

    | | value |
    |---|---:|
    | Spearman(`Photoconductance (Max)`, `Normalized photoconductance`) | **−0.5484** |
    | p | **0.0343** |
    | strongest film, 8.81e-07 (sample 15) | normalises to **0.010**, the column minimum |
    | films at exactly 1.000 | samples 4, 7 and 10, all at low raw photoconductance |

    So the axis currently rewards *weaker* photoconductance, and it is significant
    rather than noisy. **The group has flagged this and is supplying the intended
    formula.**

    **This is very likely why optoelectronic is unlearnable.** Its plain GP sits
    at −0.5842, below the null, and the first campaign's mean function makes it
    worse rather than better. No model can learn a column that ranks backwards
    against its own measurement, and half of this objective is that column.

    **Reported as a graded finding, never as a gate**, by
    `scores.AgreementCheck`: which column the model trains on is the group's
    decision, and a diagnostic that blocked a round would make that decision by
    refusing to run. It appears in the workbook read, in
    `scripts/intake_new_data.py` output, and as a standing notice on the `Review`
    sheet, so nobody reviews a batch without knowing the axis is provisional.

    **Closure path: one recipe edit plus one intake run.** Replace the
    pass-through input with the real derivation in
    `objectives.specs[1].measurement`, bump `contract_version`, re-run the intake,
    and re-decide the mean function — it may well earn its place once the column
    tracks its measurement. `test_second_campaign.py` pins the −0.5484, so that
    test failing is the signal to update the record rather than to loosen the
    check.

**Nothing on this list is now blocked on code.** What remains is a human reading a
proposed batch (issue 4), the R1 triplicates arriving (issue 7), the
photoconductance formula (issue 10), and a decision about whether an
`anneal_temp` floor belongs in `constraints:` — which is now a live list rather
than an empty one, so adding it is a two-line change.

## Are beta = 4.0 and radius = 0.25 defensible? (FIRST campaign)

> Superseded for the live campaign by "Are beta = 36 and radius = 0.35
> defensible?" near the top of this document. Kept as the record of how the first
> campaign's defaults were checked.

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

**One limit worth stating**: `radius` is not binding **on DTLZ2**. Achieved batch
spacings there are 0.72–0.98, far above every radius tested, so local penalization
rarely has two candidates close enough to penalise — visible in `beta=8` giving
identical results at radius 0.15 and 0.25. This sweep therefore validates `beta`
properly and says little about `radius` on that problem.

**It does bind on the live campaign, and the earlier claim here that it probably
did not was itself an artifact.** That claim rested on the R1 batch's minimum
spacing of 0.921 — a number produced by the mis-encoded UCB-HVI baseline described
in `docs/R1_BATCH_WITHDRAWAL.md`. With the baseline corrected the live R1 batch
spaces at 0.6337, and the round simulation measures a clean monotone staircase
(`scripts/plot_round_simulation.py`, 13 cells, seed 73, oracle-scored R0):

| radius | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| achieved R1 spacing | 0.455 | 0.455 | 0.455 | 0.543 | 0.720 | 0.921 | 0.921 | 0.921 | 0.921 |
| range-edge coords | 11 | 11 | 11 | 12 | 13 | 13 | 15 | 15 | 15 |

Nine cells produce **six distinct R1 batches**. `radius` binds below about 0.30
and saturates above it, and it buys spacing at a measurable cost: **11 → 15
range-edge coordinates across the arm.** That trade-off — diversity against
edge-pinning — had not been measured before, and it is a policy choice for the
group rather than a tuning question.

**`radius = 0.25` stays the default for now**, mid-staircase, but as a declared
choice rather than an inherited one. Note the numbers above are single-seed:
*which* batch a cell proposes is a fact, because the pipeline is deterministic at
a fixed seed, but the hypervolumes cannot rank cells at n = 1. Re-run the radius
arm at ~5 seeds before changing the default on performance grounds.

Inert is acceptable for a safety knob, but then it has to be shown to work
deliberately rather than inferred from a campaign that never exercised it.
`test_radius_pushes_the_second_pick_out_of_the_penalised_neighbourhood` does that
by construction: three candidates crowded 0.02 apart scoring better than an
isolated fourth, where greedy selection takes the two best and penalization pushes
the second pick beyond the radius. Its companion pins the inert case — a radius
smaller than the gaps must change nothing.

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

**This is the canonical instrument for LOO numbers from now on.** It reports plain
thickness at +0.116 where `GP_MODEL_DECISION.md` records +0.183; that was
reconciled on 2026-07-30 and the whole difference is the data, not the method. The
older instrument read the workbook's stored `ROUND(mean(T1..T4))`; the model now
trains on the unrounded mean. Seven of fifteen rows change, by at most 0.50 nm, and
that alone moves LOO R² by 0.067 — the pipeline contributes exactly nothing, since
with no mean function the two routes are the same code. Same fragility as the 0.089
above, and comfortably inside the ±0.236 floor. Neither conclusion changes: the
structured swing is +0.201 on the old values and +0.265 on the new.

## Reproducing the analysis

```bash
python scripts/gp_diagnostic.py --variants legacy_matern_no_prior dim_scaled_prior
python scripts/validate_structured_means.py
python scripts/thickness_objective_check.py
python scripts/dtlz2_parameter_sweep.py            # beta x radius, needs no data
```

All but the sweep need the ignored private workbook at
`local_inputs/Summary Table.xlsx`.
