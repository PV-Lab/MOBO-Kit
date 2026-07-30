# Handoff

Written 2026-07-30, at the end of the session that closed the numbered issue list.
Read this first in a new session.

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

Expect **425 passed, 0 failed** (~90 s). If that holds, everything below is true.
Two tests open a real tkinter window and drive it; they skip themselves without a
display. Nothing in the suite needs the private workbook — the tests that would use
it skip when it is absent.

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

1. **Nobody has reviewed a proposed batch yet.** The artifact exists and says what
   it should — including that the `speed_1 = 1000` corner is being skipped as
   *known and bad* because the thickness trend extrapolates confidently to its
   range edge, where the only two observations disagree with each other. Fifteen
   films is a real cost. This is a human decision and is not automated.
2. **The campaign is running on data the group calls test data.** When a corrected
   or re-measured workbook arrives, run `python scripts/intake_new_data.py
   --workbook <path>`. It re-derives the floors at the new N and gives a
   per-objective keep-or-delete verdict on each mean function. Re-measuring
   samples 12 and 8 would be the single highest-value experiment: sample 12's
   1155 nm is `ROUND(mean(1600, 709))`, and because `speed_1` is a *feature of the
   thickness mean function*, re-measuring it moves the fitted trend and therefore
   the model's belief about the whole low-speed region — not just two points.
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

Plotting is the obvious next build: `CAMPAIGN_STATUS.md` has a "For the plotting
work" section with the contour-slice recipe and the two conventions that silently
produce wrong pictures.

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
- **`beta = 4.0` and `radius = 0.25` stay**, under a rule fixed before the sweep
  ran. `radius` was *not* exercised by that sweep — DTLZ2 batches land 0.72–0.98
  apart — so it is verified separately by construction in
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

## Four facts about the tooling that will bite you

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
