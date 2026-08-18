# Handoff

Written 2026-07-30, at the end of the session that closed the numbered issue list.
Updated 2026-08-17. Read this first in a new session.

## Where to start today

**A second campaign is live on a second dataset.**
`configs/campaign_d2d_perovskite_test.yaml`, contract `d2d-objectives-v3-test`,
against `local_inputs/Summary Table Test.xlsx`. Two objectives are computed
differently, two grids moved, and this project's first real constraints are
active. `configs/campaign_d2d_perovskite.yaml` is **archived** — complete and
loadable, because every number in `GP_MODEL_DECISION.md` is about that contract,
but not something to run new rounds against. Start at the "Second campaign" section
at the top of `CAMPAIGN_STATUS.md`.

**Nothing since 2026-07-31 has been pushed, by decision.** `colin` is four commits
ahead of `origin/colin`: the SHAP attribution work, the beta x radius boxplot
sweep, the tkinter capture fix, and the second campaign. The group is keeping this
local for now, so do not push without asking.

**Proposing a round now also renders six figures**, beside the workbook under
`<stem>_reports/<round>_<UTC timestamp>/`, and a second button renders the four
that need no batch. `docs/CAMPAIGN_STATUS.md` has the "round report" section:
what each figure can and cannot claim, the two equalities its tests assert, and
the three notebook conventions deliberately not carried over.

**The thickness mean function is settled at p = 0.0028** (rank permutation, 1800
shuffles, 4 exceedances, 95% CI [0.0003, 0.0052]). Intake had left it inconclusive
on R2, which is a statement that R2 cannot resolve it at N=15 rather than a
verdict; `scripts/permutation_rank_test.py` is the instrument that decides, and
intake now says so when it lands there.

**The launcher's DEFAULT_CONFIG is a product decision, not a constant.** It is the
one path an experimentalist reaches by double-clicking, so it must track the
ACTIVE campaign. Archiving a config without moving that line points the GUI at a
retired contract, and the symptom is a missing-column error that reads as a broken
workbook. That reached a user on 2026-08-18; the default is now pinned by a test,
and a mismatch names the config, its status and the near-miss headers.

## Read these, in this order (~20 minutes)

1. **`README.md`** — what the toolkit is, the three-round loop, current parameters,
   and how an experimentalist runs a round without writing code.
2. **`docs/CAMPAIGN_STATUS.md`** — the working guide: what to pass, what comes
   back, how the objectives are computed, and the numbered issues with their
   evidence. Longest of the three and the one to keep open while working.
3. **`docs/GP_MODEL_DECISION.md`** — why the model is the way it is. Every claim
   has a measured number. Read it before changing the model; skip on a first pass
   if you only need to *use* the toolkit.

Then verify the state yourself:

```bash
pytest -q
```

Expect **578 passed, 0 failed, 28 warnings** (~180 s). If that holds, everything
below is true. Six tests open a real tkinter window and drive it; they skip
themselves when Tk will not start. Nothing in the suite needs a private workbook —
the tests that would use one skip when it is absent.

**`--capture=sys` in `addopts` is load-bearing, not a preference.** pytest's
default fd-level capture swaps file descriptors 1 and 2, and a Tk interpreter
built while that is in force holds descriptors that are gone by the time the next
one is built — so the second or third launcher window in a process dies reading
its own `init.tcl` and reports `No error`. It read as a race in the launcher's
stale-reply handling for a while and is neither a race nor a launcher defect.
Measured: 6 failures in 9 runs of one launcher test under `--capture=fd`, none
under `--capture=sys`. Only `capsys` is used in this suite, never `capfd`. The
`open_window` fixture in `tests/test_launcher.py` carries the full account.

## Where the project stands

The loop is complete and runs end to end, from a spreadsheet an experimentalist
fills in to a batch of conditions with a review artifact attached.

```
Summary Table.xlsx
   -> scores.py            objectives computed from raw measurements,
                           stored score cells demoted to warning cross-checks
   -> campaign.py          GP per objective with a physics-informed mean,
                           UCB-HVI (R1) or qLogNEHVI (R2), local penalization
   -> batch_review.py      predictions in physical units, uncertainties,
                           distance to observed, range-edge coordinates, probes
   -> workbook_io.py       worklist + Review sheet written BESIDE the workbook
   -> back again           read_candidate_results aggregates films to design
                           points so R1 -> R2 can advance
```

`launch_mobo_kit.bat` / `.command` puts a small window over that. The acquisition
modules are still byte-identical to where this project started: `ucb_hvi.py`,
`qlognehvi_batch.py`, `batch_selection.py`, `discrete_refinement.py`,
`candidate_pool.py`, `sobol_pool.py`, `lhs.py`, `design.py`. What changed is the
GP's priors and guards, the objective layer, and orchestration on top.

Verified without any experimental data, on DTLZ2, whose Pareto front is known:

```bash
pytest tests/test_dtlz2_acceptance.py -m "not slow"   # structure
pytest tests/test_dtlz2_acceptance.py -m slow         # BO vs random
python scripts/plot_dtlz2_report.py                   # the figures
python scripts/dtlz2_parameter_sweep.py               # beta x radius
```

## What is actually open

Everything numbered in `CAMPAIGN_STATUS.md` is closed, wired-and-waiting, or needs
a person rather than code. In rough priority:

1. **Nobody has reviewed a proposed batch yet, and the batch was reissued on
   2026-07-31.** The first one was withdrawn: `run_r1_ucb` had been scoring
   candidates against an observed baseline whose thickness axis collapsed to zero
   (`docs/R1_BATCH_WITHDRAWAL.md`, and issue 9 in `CAMPAIGN_STATUS.md`). Four of
   the five conditions survived; one was replaced. **No films were fabricated from
   the withdrawn batch** — the review gate did exactly what it exists for.
   The reissued sheet is `local_inputs/Summary Table_R1_Candidates.xlsx`, and it
   still says what it should, including that the `speed_1 = 1000` corner is being
   skipped as *known and bad* because the thickness trend extrapolates confidently
   to its range edge, where the only two observations disagree with each other.
   Fifteen films is a real cost. This is a human decision and is not automated.
2. **The campaign is running on data the group calls test data, and workbook or
   formula changes are anticipated.** When a corrected or re-measured workbook
   arrives, run `python scripts/intake_new_data.py --workbook <path>`. It
   re-derives the floors at the new N and gives a per-objective keep-or-delete
   verdict on each mean function.

   **Route any change through the config, not through code.** A changed
   measurement column or a changed score formula is a `objectives.specs[].measurement`
   recipe edit plus a bump of `objectives.contract_version`, followed by one intake
   run. The version bump is the part people skip: utility space is what
   hypervolume is measured in, so a silently redefined objective makes every
   cross-round number incomparable while every plot still renders. `scores.py`
   recomputes from raw columns and cross-checks the stored cells, so a stale pasted
   literal announces itself rather than propagating.

   **Note what the group's 2026-07-31 decision did to the re-measurement case.**
   Re-measuring samples 12 and 8 was previously the highest-value experiment on the
   grounds that sample 12's 1155 nm — `ROUND(mean(1600, 709))` — had a weak claim
   to being one measurement. The group has now confirmed that variation is *real*
   and the mean is the intended summary, so re-measuring would reproduce the spread
   rather than resolve it. What survives is the narrower version already on record:
   collect **more thickness points per film** in the low-speed region, not more
   films. The leverage fact is unchanged — `speed_1` is a feature of the thickness
   mean function, so that region still moves the fitted trend rather than one point.
3. **Phase 4 needs the R1 triplicates.** `replicate_variance.py` is wired and
   tested against synthetic replicates; enabling it is one config key
   (`model.observation_noise: replicate_pooled`).
4. **`anneal_temp` sits at its range edge in every proposed condition**, which is
   the monotone mean function speaking rather than a discovery. If the group would
   never anneal below some temperature, that belongs in `constraints:` — currently
   empty — and is much better learned now than after a batch ships.
5. **One statistical question is still open**: does linear-mean-plus-GP beat
   linear-mean-alone under the permutation null? It was always meant to ride along
   with the optoelectronic permutation run and gates nothing.

Plotting landed on 2026-07-31: `scripts/plot_round_simulation.py` runs the whole
loop against a frozen GP oracle and renders it, and `CAMPAIGN_STATUS.md` still has
the "For the plotting work" section with the contour-slice recipe and the two
conventions that silently produce wrong pictures. Read `docs/ROUND_SIM_DELTA.md`
before extending it — the approach came from Annie Xu's fork and the delta list
records what was changed and why.

## Three floors. Check all three before comparing any two numbers.

- **Null, −0.148 at N=15.** Predicting the leave-one-out mean gives
  `1 − (N/(N−1))²`. A model below it learned nothing; negative LOOCV Spearman is
  the signature, not a sign bug. It moves with N — recompute rather than reuse.
- **Sampling, ±0.236.** Parametric bootstrap, 4000 resamples. Two LOO R² values
  less than about half a point apart are not a comparison at this N.
- **Numerical reproducibility, ≈0.07** (new, 2026-07-30). Two independent
  perturbations that change nothing meaningful each move LOO R² by that much: the
  MLL optimiser landing elsewhere on an *identical* likelihood surface (0.0715),
  and rounding thickness to whole nanometres, ≤0.5 nm on 7 of 15 rows (0.0670).
  A second-decimal difference is not a measurement.

This project argued inside a floor twice, both times because the number moved in
the pleasing direction. Derivations and evidence are in `GP_MODEL_DECISION.md`.

## Settled, do not reopen

- **Sample 1 stays in.** Dropping it improves the fit, but sample 12 improves it
  nearly twice as much and has the highest leverage in the design — a
  high-leverage-endpoint artifact, not a provenance signal.
- **`speed_2` does not explain the low-speed contradiction.** Adding
  `log(speed_2 + 1)` to the thickness mean drops LOO R² from +0.449 to −0.827.
- **Uniformity has no learnable signal** from these 15 rows — nothing beat the null
  across ~240 model configurations, permutation p = 0.82. Exploration-only.
  Whether that is physics or measurement noise is answerable from the R1
  replicates and not before.
- **The 0.089 optoelectronic gap is explained**, not merely bounded: the two
  pipelines specify the same model, about a fifth of the gap is the outcome
  transform standardizing different quantities, and the rest is the optimiser.
- **`beta = 4.0` stays**, under a rule fixed before the DTLZ2 sweep ran.
  **`radius = 0.25` also stays, but its justification changed on 2026-07-31 and it
  is now a declared policy choice rather than a settled one.** DTLZ2 never
  exercised the knob (batches land 0.72–0.98 apart), and the claim that the live
  campaign was the same rested on a spacing figure the R1 baseline defect had
  inflated. On the corrected landscape `radius` binds below about 0.30, staircases
  achieved spacing 0.455 → 0.921, and trades diversity against range-edge pinning
  (11 → 15 edge coordinates). See `CAMPAIGN_STATUS.md`, "Are beta = 4.0 and
  radius = 0.25 defensible?". The mechanism is still verified by construction in
  `tests/test_batch_selection.py`.
- **The thickness mean function's evidentiary weight is the rank permutation**
  (p = 0.0350, 95% CI [0.0270, 0.0446] at 1800 shuffles). The R² swing is
  consistent with it and no more.

## Instruments: which number came from what

`scripts/intake_new_data.py` is **canonical** for LOO numbers. Where it disagrees
with `GP_MODEL_DECISION.md`, it is right and that document is historical: it read
the workbook's rounded `Thickness (avg)` while the model now trains on the
unrounded mean. The one visible disagreement — plain thickness +0.183 against
+0.116 — is entirely that, and no conclusion depends on it.

## The one that actually bit, and the shape it shares with two others

**`ObjectiveTransform.transform` takes MODEL-space values, not measurements.** It
decodes the link itself — `exp()` for a log objective — so handing it thickness in
nanometres exponentiates a value that was never a logarithm. `exp(360…1303)`
saturates the 650 nm Gaussian to exactly `0.0`, which is finite, so nothing
raises. `run_r1_ucb` did this to its observed HVI baseline for the life of the
campaign: baseline hypervolume **0.004659 against a true 0.436442**, and one of
five proposed conditions was an artifact of it. Use
`transform.transform_measurements` at any call site holding workbook values.

It is the **third plausible-finite-number failure** here, after the hypervolume
auto-reference and the swallowed `train_Yvar`. All three were finite,
ordinary-looking wrong answers compared against nothing. The defence that works is
not another guard — each passed every guard it met — it is making the quantity
observable and reproducing it by a second route. If you add a number that steers a
decision, add the comparator with it.

## Four more facts about the tooling that will bite you

- **openpyxl discards cached formula values on save.** Verified: `Z2:Z4` read
  `[0.657, 0.587, 0.561]` before a save that only added an empty sheet, and
  `[None, None, None]` after. This is why `workbook_io` writes to a *sibling file*
  and never opens the source for writing. Do not "simplify" it.
- **BoTorch's `Hypervolume` assumes maximisation and silently drops points that do
  not dominate the reference.** No warning, no exception — a smaller number, or
  0.0. `metrics.compute_ref_pareto_hv` now refuses that case.
- **BoTorch silently ignores `train_Yvar` when a `likelihood` is also passed.** The
  likelihood wins, stays single-element, and the replicate information vanishes
  with no error. Verified on 0.15.1. Pass one or the other, never both.
- **`Standardize` rescales `train_Yvar` along with the targets**, so measured
  variance must arrive in the target's own units — and in the *model's* space,
  which for thickness is `log T`, not nanometres.

## Where the removed history went

The Step 1/2A/2B/2C audit apparatus (50 files, 16,442 lines) was removed on
2026-07-29 once its findings were recorded. Recover any of it with:

```bash
git show pre-cleanup-2026-07-29:src/mobo_kit/<file>.py
git show 19591cc:docs/D2D_CAMPAIGN_SPEC.md
```

`d2d_scores.py` in that tag is the objective-from-raw-columns computation with the
polarity inverted — it treated the stored score cells as authoritative. It became
`scores.py`, the other way round.

## The notebook is a demo, not the campaign path

`notebooks/MOBO_demo_annotated.ipynb` runs the general toolkit API over an
arbitrary CSV. Every import in it still resolves and no call has drifted, but it
builds its GPs with `models.fit_gp_models` — the **prior-free** construction that
`GP_MODEL_DECISION.md` documents as degenerate on small data (ARD lengthscales
0.13 to 38,000, noise pinned at its floor). Anyone following it as "the recommended
way" would rebuild the model this project replaced. A scope note at the top of the
notebook now says so and points at `campaign.fit_campaign_models`.

`models.py` is kept because the notebook and the older `main.py` path use it. It is
legacy, not dead, and it is not what a round runs.

## Working advice

Develop against **DTLZ2**, not the campaign workbook. The group has described the
current numbers as test data, and several turns of this project went into forensics
on 15 rows that may be replaced. The synthetic path gives a known answer to score
against, and anything data-specific lives in config — so a new dataset means a new
YAML, not new code.

Two process rules this project learned the hard way, both worth keeping:
**verification gates the commit** — run the tests as their own step, never in the
same breath as `git commit` — and **an order-dependent or timing-sensitive test
failure is a real defect until proven otherwise**, in the test or in the product.
Both of those cost a commit-with-a-red-suite before they were adopted.
