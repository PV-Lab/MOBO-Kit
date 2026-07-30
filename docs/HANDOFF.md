# Handoff

Written 2026-07-29 at the end of the session that fixed the GP and cleaned the
branch. Read this first in a new session.

## Read these three files, in this order (~15 minutes)

1. **`README.md`** — what the toolkit is, the three-round loop, current parameters.
2. **`docs/CAMPAIGN_STATUS.md`** — how to call it, what comes back, how to plot
   it, and the numbered open issues. This is the working guide.
3. **`docs/GP_MODEL_DECISION.md`** — why the model is the way it is. Every claim
   has a measured number attached. Skip on a first pass if you only need to *use*
   the toolkit; read it before changing the model.

Then verify the state yourself in one command:

```bash
pytest -q
```

Expect **399 passed, 0 failed** (~90 s). If that holds, everything below is true.

Two of those tests open a real tkinter window and drive it; they skip themselves
if there is no display.

## What works

The full loop runs end to end: `run_r0_lhs` → `run_r1_ucb(5)` → `run_r2_qlognehvi(3)`,
23 distinct conditions, three replicate films each. Verified on DTLZ2 — a
synthetic problem with a known Pareto front — so the algorithm is checkable
without any experimental data:

```bash
pytest tests/test_dtlz2_acceptance.py -m "not slow"   # structure
pytest tests/test_dtlz2_acceptance.py -m slow         # BO vs random
python scripts/plot_dtlz2_report.py                   # the figures
```

**The acquisition code was never modified.** `ucb_hvi.py`, `qlognehvi_batch.py`,
`batch_selection.py`, `discrete_refinement.py`, `candidate_pool.py`,
`sobol_pool.py`, `lhs.py`, `design.py` are byte-identical to the pre-session
state. What changed is the GP (priors, guards), the objective layer (how a
measurement becomes a utility), and new orchestration on top.

## Open issues, in the order I would work them

1. ~~Wire the launcher.~~ **Done 2026-07-30.** `launcher.py` plus
   `launch_mobo_kit.bat` / `.command`. The handoff said this had "no dependency on
   anything else here" and that was wrong: nothing read a filled-in candidate
   sheet back, so R1 → R2 could not advance. `workbook_io.read_candidate_results`
   now does, aggregating each condition's films to one observation — thickness in
   log space, matching what the GP trains on. Details in `CAMPAIGN_STATUS.md`,
   "Reading a round's results back". The window itself is a shell over
   `inspect_campaign` / `gather_observations` / `generate_next_round`, which are
   tested headlessly.
2. ~~Compute the three objectives from the measurement columns.~~ **Done
   2026-07-30.** `src/mobo_kit/scores.py` computes them from `Coverage`/
   `Uniformity`/`Phase purity`, `PL`/`Photoconductance` and `T1..T4`; the stored
   score cells are now cross-checks that warn. The audit that motivated it, the
   agreement numbers, and why `Y`/`AB` are deliberately *not* cross-checked are in
   `CAMPAIGN_STATUS.md` issue 2. Two consequences worth carrying forward:
   thickness now reaches the GP **unrounded** (663.75 rather than 664), and the
   R1 candidate sheet asks for raw measurements instead of derived scores, so an
   R1 sheet generated before this date has the wrong columns — regenerate it.
3. **The unexplained 0.089 on optoelectronic.** Two implementations of one
   pipeline on the same 15 rows give LOO R² +0.355 and +0.267. Ruled out: the
   mean feature, sampling noise, the standardization scale. Untested: MLL
   optimiser seeding, and the residual-vs-target training interaction. Both
   numbers beat plain (−0.342), so the direction is safe; close the gap before
   acting on optoelectronic candidates.
4. **The review artifact is built (2026-07-30); the human review is still owed.**
   `batch_review.py` writes a `Review` sheet beside the worklist and echoes it into
   the launcher. What it reported on the R0-trained batch, and why the
   `speed_1 = 1000` avoidance turned out to be the mean function extrapolating
   rather than a local average, is in `CAMPAIGN_STATUS.md` issue 4. Someone still
   has to read it and decide — that part is not automatable and is not automated.
5. **Replicate variance into `train_Yvar`** (needs R1 measurements, so it is
   gated on the batch shipping). Both config decisions are now made and recorded:
   thickness variance pools in **log space**, and
   `read_candidate_results().replicate_spread` already returns it there. What is
   left is passing it to the model and deciding the R0 policy — the R0 rows have
   no replicate films, but they do have 2-4 thickness points each, pooling to a
   within-row sd of `log T` of 0.244 over 24 dof. That is within-film spread, not
   film-to-film, so it is a floor rather than an estimate.
6. ~~`metrics.compute_ref_pareto_hv` has a degenerate auto-reference.~~ **Done
   2026-07-30.** The `ref_point_np=None` path is gone: a missing reference now
   raises and names `reference_point_utility`. The same function also refuses a
   reference nothing dominates, rather than reporting the 0.0 that BoTorch's
   silent point-dropping would produce. `main.py` already passed an explicit
   reference and is unaffected; the demo notebook had one bare call, now fixed.
   Note for whoever reads the old issue text: `mins - 1e-8` is harmless while a
   *dominated* point sets the per-objective minima, and collapses as soon as the
   Pareto set itself does — which is what a real trade-off front looks like.
   `tests/test_metrics.py` pins that condition; there were **no tests at all** on
   this function before, which is how it survived.

## Things not to redo

**Two resolution floors. Check both before comparing any two numbers.**

- Null LOO R² at N=15 is **−0.148** (predicting the leave-one-out mean). A model
  below that learned nothing. Negative LOOCV Spearman is the signature, not a
  sign bug.
- Resolution sd is **±0.236** (parametric bootstrap, 4000 resamples). Two LOO R²
  values less than about half a point apart **are not a comparison at this N.**

This project argued inside that floor twice — once over −0.017 vs −0.145, once
over +0.355 vs +0.244 — both times because the number moved in the pleasing
direction. The floors are in `GP_MODEL_DECISION.md` for exactly this reason.

**Settled, do not reopen:**

- **Sample 1 stays in.** Dropping it improves the fit, but sample 12 improves it
  nearly twice as much and has the highest leverage in the design. The gain is a
  high-leverage-endpoint artifact, not a provenance signal. Full reasoning and
  the leverage table are in `GP_MODEL_DECISION.md`.
- **`speed_2` does not explain the low-speed contradiction.** Adding
  `log(speed_2 + 1)` to the thickness mean drops LOO R² from +0.449 to −0.827.
- **Uniformity has no learnable signal** from these 15 rows — nothing beat the
  null across ~240 model configurations, permutation p = 0.82. Exploration-only.
  Whether that is physics or measurement noise is answerable from the R1
  replicates, and not before.

## Two facts about the tooling that will bite you

- **openpyxl discards cached formula values on save.** Verified: `Z2:Z4` read
  `[0.657, 0.587, 0.561]` before a save that only added an empty sheet, and
  `[None, None, None]` after. This is why `workbook_io` writes candidates to a
  *sibling file* and never opens the source for writing. Do not "simplify" it.
- **BoTorch's `Hypervolume` assumes maximisation and silently drops points that
  do not dominate the reference.** No warning, no exception — just a smaller
  number, or 0.0. Assert at least one point dominates before trusting it.

## Where the removed history went

The Step 1/2A/2B/2C audit apparatus (50 files, 16,442 lines) was removed on
2026-07-29 once its findings were recorded. It validated a model that has since
been replaced. Recover any of it with:

```bash
git show pre-cleanup-2026-07-29:src/mobo_kit/<file>.py
git show pre-cleanup-2026-07-29:docs/STEP2C_ROBUSTNESS_HANDOFF.md
```

Two files there are worth knowing about rather than rediscovering:

- `src/mobo_kit/d2d_scores.py` — the three objective formulas computed from raw
  measurement columns, with per-row tolerance comparison and warning/error
  severities. It is open issue 2 already written, with the polarity inverted.
- `docs/D2D_CAMPAIGN_SPEC.md` — the Step 1 data contract, deleted 2026-07-29.
  Its input grid and on-grid rules now live in `configs/` and are enforced by
  `design.py` and `campaign.validate_batch`; its thirteen "unresolved decisions
  blocking real R1" are resolved in the config; and its workbook audit described
  a revision of `Summary Table.xlsx` that no longer matches the file (it reports
  duplicate `Uniformity score` headers at Q/T, which the current workbook does
  not have). Recover with `git show 19591cc:docs/D2D_CAMPAIGN_SPEC.md`.

## Working advice

Develop against **DTLZ2**, not the campaign workbook. The experimental group has
described the current numbers as test data, and several turns of this project went
into forensics on 15 rows that may be replaced. The synthetic path gives a known
answer to score against, and anything data-specific lives in config — so a new
dataset means a new YAML, not new code.
