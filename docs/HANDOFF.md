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

Expect **280 passed, 0 failed** (~105 s). If that holds, everything below is true.

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

1. **Wire the launcher.** `workbook_io.py` has the reader, round detection and
   candidate writer. What is missing is the double-click `.bat` / `.command` and
   the small tkinter window. No dependency on anything else here.
2. **Compute the three objectives from the literal measurement columns**
   (L/N/O, P/Q, X) and demote the formula columns to a cross-check that warns on
   disagreement. This matters: column AA is a **pasted literal, not a formula**,
   so it silently will not update if P or Q are edited — the same failure that
   produced the original uniformity discrepancy. Audit AB too.
3. **The unexplained 0.089 on optoelectronic.** Two implementations of one
   pipeline on the same 15 rows give LOO R² +0.355 and +0.267. Ruled out: the
   mean feature, sampling noise, the standardization scale. Untested: MLL
   optimiser seeding, and the residual-vs-target training interaction. Both
   numbers beat plain (−0.342), so the direction is safe; close the gap before
   acting on optoelectronic candidates.
4. **Human review of a proposed batch.** Print the five conditions in physical
   units with predicted objectives, uncertainties, and distance to the nearest R0
   point. Fifteen films is a real cost and nobody has looked yet. One thing to
   watch: whether anything lands near `speed_1 = 1000`, a region holding two
   observations that contradict each other.
5. **Replicate variance into `train_Yvar`** (needs R1 measurements, so it is
   gated on the batch shipping). Two decisions to make in config *now*, not when
   the data arrives: pool thickness variance in **log space** (`response: log`
   means the GP trains on log T), and decide the policy for the R0 rows, which
   have no replicates.
6. **`metrics.compute_ref_pareto_hv` has a degenerate auto-reference** —
   `mins - 1e-8` gives HV 6e-8 against 1.448 from BoTorch's
   `infer_reference_point`. Only the `ref_point_np=None` path. Any new plotting
   code must pass a fixed reference, or an HV-vs-round curve is meaningless.

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

## Working advice

Develop against **DTLZ2**, not the campaign workbook. The experimental group has
described the current numbers as test data, and several turns of this project went
into forensics on 15 rows that may be replaced. The synthetic path gives a known
answer to score against, and anything data-specific lives in config — so a new
dataset means a new YAML, not new code.
