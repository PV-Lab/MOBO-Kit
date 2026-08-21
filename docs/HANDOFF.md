# Handoff

Read this first in a new session. Updated 2026-08-18, at the audit pass before the
first push since 2026-07-31.

## What this repository is doing right now

**There are two campaigns, and only one of them is real.**

| | first — algorithm testing | second — **the live campaign** |
|---|---|---|
| config | `configs/campaign_d2d_perovskite.yaml`, **archived** | `configs/campaign_d2d_perovskite_test.yaml` |
| contract | `d2d-objectives-v2-nm-thickness` | `d2d-objectives-v3-test` |
| workbook | `local_inputs/Summary Table.xlsx` | `local_inputs/Summary Table Test.xlsx` |
| purpose | proving the loop worked | the experiment being run |

The first campaign's numbers were how the toolkit was validated. **Two of the three
objectives are computed differently in the second**, so none of those fitted
numbers transfer — they describe quantities that were redefined. Every document
that is about the first campaign now says so in a banner at the top. Everything
else is about the live one.

The launcher, `intake_new_data.py`, `permutation_rank_test.py`,
`generate_round_report.py`, `plot_round_simulation.py` and `plot_boxplot_sweep.py`
all default to the live config. **Archiving a config without moving the launcher's
default is how a user got a missing-column error on an intact workbook**; a test
now pins that the launcher's default names an active campaign.

Both workbooks live under `local_inputs/`, which is gitignored and never travels
by git. Copy them by hand on any move.

## Read these, in this order (~25 minutes)

1. **`README.md`** — what the toolkit is, the two campaigns, the three-round loop,
   how an experimentalist runs a round without writing code, and how `beta` and
   `radius` were chosen.
2. **`docs/CAMPAIGN_STATUS.md`** — the working guide and the longest of the three.
   Start at "Second campaign", which is at the top; everything below that section
   describes the first campaign unless it says otherwise.
3. **`docs/GP_MODEL_DECISION.md`** — why the model is the way it is. It is the
   **first** campaign's record and carries a banner saying so. What still applies
   is the *method* — the floors, the null, refitting a trend inside every fold, the
   two degenerate fitting modes — and none of its LOO numbers.

Then verify the state yourself:

```bash
pytest -q
```

Expect **580 passed, 0 failed, 28 warnings** (~180 s). Nothing in the suite needs a
private workbook; the tests that would use one skip when it is absent.

**`--capture=sys` in `addopts` is load-bearing, not a preference.** pytest's
default fd-level capture swaps file descriptors 1 and 2, and a Tk interpreter built
while that is in force holds descriptors that are gone by the time the next one is
built — so the second or third launcher window in a process dies reading its own
`init.tcl` and reports the unhelpful message `No error`. It read as a race in the
launcher for a while and is neither a race nor a launcher defect. Measured: 6
failures in 9 runs of one launcher test under `--capture=fd`, none under
`--capture=sys`. Only `capsys` is used in this suite, never `capfd`. The
`open_window` fixture in `tests/test_launcher.py` carries the full account.

## The instruments, and the one command each

```bash
# audit new or corrected data, and re-decide every mean function on it
python scripts/intake_new_data.py --workbook "local_inputs/Summary Table Test.xlsx"

# adjudicate a mean function on RANK when R2 cannot resolve it
python scripts/permutation_rank_test.py --objective thickness --permutations 1800

# the six figures a round produces, from a terminal instead of the button
python scripts/generate_round_report.py --workbook "local_inputs/Summary Table Test.xlsx"

# the campaign loop against a frozen oracle, at the ratified knobs
python scripts/plot_round_simulation.py --workbook "local_inputs/Summary Table Test.xlsx" --cell 0.35,36
```

`launch_mobo_kit.bat` / `.command` is the one-button path: check the workbook,
propose the next round, and get the figures. It writes a worklist and a `Review`
sheet **beside** the workbook and never into it.

**There is one leave-one-out fold loop, `mobo_kit.loocv`, and three callers share
it.** Intake is canonical for LOO numbers, the round report plots them, and the
permutation test builds a null out of them. They were briefly three
implementations; a test now asserts they are the same function object rather than
that they agree.

## Where the live campaign stands

Measured on its 15 rows by `intake_new_data.py`, against a leave-one-out null of
**−0.1480**:

| objective | plain GP | with mean function | verdict |
|---|---:|---:|---|
| uniformity | −0.6447 | — | below the null → **exploration only** |
| optoelectronic | −0.5842 | −0.6977 | below the null → **exploration only**, mean function deleted |
| thickness | +0.5227 | +0.6630 | **learnable** |

**Two of the three axes carry no signal.** A batch is therefore chosen on one
informative axis and two uninformative ones. That is a legitimate exploration
round; it is not a three-objective optimisation, and the review says so rather
than letting the predicted numbers imply otherwise.

**Thickness keeps its mean function on the rank permutation**, not on R²:
observed rank ρ **+0.7250**, null mean −0.1917 (sd 0.2937), **4 exceedances in
1800**, **p = 0.0028, 95% CI [0.0003, 0.0052]**. Intake had left it *inconclusive
on R²* — the +0.1403 swing sits inside the ±0.236 floor — which is a statement
that R² cannot resolve it at N=15 rather than a verdict. Rank is what the
acquisition consumes; it never sees R². **Do not quote the swing as evidence.**

**Knobs: `beta = 36`, `radius = 0.35`**, a declared policy about how much to
explore rather than a measured optimum. Two consequences are on record — local
penalization is inert (batch spacing 1.091 against a 0.35 radius) and the batch
runs to the edges (21 of 50 coordinates at a bound) — along with two triggers for
revisiting. `CAMPAIGN_STATUS.md` has the section.

## What is actually open

1. **Nobody has run a batch on the live campaign yet.** No worklist exists for it;
   pressing **Propose R1** writes one, plus the Review sheet and six figures.
   Fifteen films is a real cost, and whether to fabricate is a human decision that
   is not automated.
2. **The photoconductance normalisation is wrong and the group is fixing it**
   (issue 10). `Normalized photoconductance` does not rank like the raw
   measurement it summarises: Spearman **−0.5484**, p = 0.0343, with the strongest
   film carrying the column minimum. It is half of the optoelectronic objective
   and is the prime suspect for that axis being unlearnable. Closing it is one
   recipe edit plus one intake run — and the mean function should be re-decided
   afterwards, because it may well earn its place once the column tracks its
   measurement.
3. **Phase 4 waits on the R1 triplicates.** `replicate_variance.py` is wired and
   tested; enabling it is one config key, `model.observation_noise:
   replicate_pooled`.
4. **`anneal_temp` sits at a range edge in proposed conditions.** If the group
   would never anneal below some temperature, that belongs in `constraints:` —
   which is now a live list with three entries, so adding one is a two-line change.

## Three floors. Check all three before comparing any two numbers.

These are method, and they carry across both campaigns.

- **Null, −0.148 at N=15.** Predicting the leave-one-out mean gives
  `1 − (N/(N−1))²`. A model below it learned nothing, and a negative LOOCV
  Spearman is that signature rather than a sign bug. It moves with N — recompute.
- **Sampling, ±0.236.** Parametric bootstrap, 4000 resamples at N=15. Two LOO R²
  values less than about half a point apart are not a comparison at this N.
- **Numerical reproducibility, ≈0.07.** Two perturbations that change nothing
  meaningful each move LOO R² by that much. A second-decimal difference is not a
  measurement.

**When a comparison lands inside a floor, that is not a verdict — it is a
statement that the instrument cannot decide, and a different instrument should.**
For a mean function that instrument is the rank permutation. This project argued
inside a floor twice before adopting that rule.

## Settled, do not reopen

- **The two campaigns' objectives are different quantities.** A shared
  `contract_version` would make their hypervolumes look comparable when they
  measure different spaces. That is why the second campaign is a new config file
  rather than an edit.
- **`ObjectiveTransform.transform` takes MODEL-space values, not measurements.**
  It decodes the link itself, so handing it thickness in nanometres exponentiates
  a value that was never a logarithm. Use `transform.transform_measurements` at
  any call site holding workbook values. This defect has now arrived by three
  separate routes; the third was caught in a draft of the permutation script only
  because saturating the Gaussian to 0.0 made a column constant.
- **Uniformity has no learnable signal** on either campaign's data — the first by
  permutation (p = 0.82 on *that* score), the second by leave-one-out (−0.6447).
  Exploration-only by measurement, not by choice.
- **openpyxl discards cached formula values on save**, which is why candidate
  sheets are written to a *sibling file* and the source workbook is never opened
  for writing. Do not "simplify" that.

## The failure shape that keeps recurring

**A wrong answer that is finite, ordinary-looking, and compared against nothing.**
Four instances so far: the hypervolume auto-reference, the silently swallowed
`train_Yvar`, the R1 baseline mis-encoding, and a timing measured under CPU
contention that nearly shipped as a documented number.

No guard catches these — each passed every guard it met. What works is **making
the quantity observable and reproducing it by a second route**. So: every figure
in a round report writes the CSV behind it; the parity numbers are literally
intake's function; the batch figure reads the Review artifact rather than
recomputing it; `validate_batch` re-checks constraints the candidate pool already
filtered. If you add a number that steers a decision, add its comparator with it.

## Four tooling facts that will bite you

- **BoTorch's `Hypervolume` assumes maximisation and silently drops points that do
  not dominate the reference.** No warning, no exception — a smaller number, or
  0.0. `metrics.compute_ref_pareto_hv` refuses that case and requires an explicit
  reference.
- **BoTorch silently ignores `train_Yvar` when a `likelihood` is also passed.**
  Verified on 0.15.1. Pass one or the other, never both.
- **`Standardize` rescales `train_Yvar` along with the targets**, so measured
  variance must arrive in the target's own units — and in the *model's* space,
  which for thickness is `log T`, not nanometres.
- **`tight_layout` does not support 3-D axes or colorbars** and warns that its
  result may be wrong. `round_report._save` takes `tight=False` for those figures
  rather than ignoring the warning.

## Working advice

Develop against **DTLZ2** where you can: `tests/test_dtlz2_acceptance.py` runs the
whole loop on a synthetic problem with a known Pareto front, so the algorithm can
be checked with no dependence on whether the measurements are right. Anything
data-specific lives in config, so a new dataset means a new YAML, not new code.

Two process rules this project learned the hard way, both worth keeping:
**verification gates the commit** — run the tests as their own step, never in the
same breath as `git commit` — and **an order-dependent or timing-sensitive test
failure is a real defect until proven otherwise**, in the test or in the product.

And one learned at the audit: **a number measured under load is an unreproduced
number.** Re-measure on an idle machine before writing it down.
