# Handoff

Read this first in a new session. Updated 2026-09-10, when the R1 films came back.

## What this repository is doing right now

**There are three objective contracts, and only the last one is real.**

| | v2 — test data | v3 — test data | v4 — **the real campaign** |
|---|---|---|---|
| config | `campaign_d2d_perovskite.yaml` (archived) | `campaign_d2d_perovskite_test.yaml` (archived) | `campaign_d2d_perovskite_final.yaml` |
| contract | `d2d-objectives-v2-nm-thickness` | `d2d-objectives-v3-test` | `d2d-objectives-v4-final-nomean` |
| workbook | `Summary Table.xlsx` | `Summary Table Test.xlsx` | `Final Summary Table.xlsx` |
| sheet | `Sheet1` | `Sheet1` | `R0` |
| purpose | early toolkit testing | rehearsing this contract's shape | **the experiment being run** |

v2 proved the loop worked. v3 rehearsed the shape of this contract on a workbook
literally called "Test". **v4 is the campaign that produces films.** Uniformity
and optoelectronic have been renormalised twice since v2, so none of the earlier
fitted numbers transfer; every document about an earlier contract carries a banner
saying so. v4 was `d2d-objectives-v4-final` until 2026-09-06; the `-nomean`
suffix records that the thickness prior was withdrawn that day.

**In v4 those two objectives are read as stored, permanently: the score value is
the interface.** How the group scores uniformity and optoelectronic is its own
method, and another group may score differently, so the toolkit takes the number
from the workbook and never re-derives it (the group's decision, 2026-09-06; it
replaces the earlier "read the value while the definitions are being revised").
That is a deliberate reversal of this project's usual polarity and it removes a
cross-check; `formula_fingerprint` is the partial replacement. It notices a
changed *definition* rather than a stale *value*, and it applies to the `R0` sheet
only, because R1 and R2 results are typed into the worklist as values. Thickness
is still computed.

**The workbook's sheet is now `R0`**, not `Sheet1`, so the source sheet is a config
key (`campaign.source_sheet`) rather than a constant. The workbook also carries an
`R1` sheet; it is deliberately not read. The round contract is unchanged — each
round's worklist goes to a NEW file beside the workbook and the source is never
opened for writing.

The launcher, `intake_new_data.py`, `permutation_rank_test.py`,
`generate_round_report.py`, `plot_round_simulation.py` and `plot_boxplot_sweep.py`
all default to v4. **Archiving a config without moving the launcher's default is
how a user once got a missing-column error on an intact workbook**; a test pins
that the launcher's default names an active campaign.

All workbooks live under `local_inputs/`, which is gitignored and never travels by
git. Copy them by hand on any move.

## Read these, in this order (~25 minutes)

1. **`README.md`** — what the toolkit is, the three contracts, the three-round loop,
   how an experimentalist runs a round without writing code, and how `beta` and
   `radius` were chosen.
2. **`docs/CAMPAIGN_STATUS.md`** — the working guide and the longest of the three.
   Its live-campaign section is at the top; everything below the divider
   describes an earlier contract.
3. **`docs/GP_MODEL_DECISION.md`** — why the model is the way it is. It is **v2's**
   record and carries a banner saying so. What still applies
   is the *method* — the floors, the null, refitting a trend inside every fold, the
   two degenerate fitting modes — and none of its LOO numbers.

Then verify the state yourself:

```bash
pytest -q
```

Expect **630 passed, 0 failed, 30 warnings** (measured 2026-09-10; about 6.5 min
with other work sharing the machine, so re-time on an idle one before quoting a
duration). Nothing in the suite needs a private workbook; the tests that would use
one skip when it is absent.

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
python scripts/intake_new_data.py --workbook "local_inputs/Final Summary Table.xlsx"

# adjudicate a mean function on RANK when R2 cannot resolve it
python scripts/permutation_rank_test.py --objective thickness --permutations 1800

# the six figures a round produces, from a terminal instead of the button
python scripts/generate_round_report.py --workbook "local_inputs/Final Summary Table.xlsx"

# the campaign loop against a frozen oracle, at the ratified knobs
python scripts/plot_round_simulation.py --workbook "local_inputs/Final Summary Table.xlsx" --cell 0.25,4
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

Measured on v4's 15 rows by `intake_new_data.py`, against a leave-one-out null of
**-0.1480** and a resolution floor of **+-0.236**:

| objective | plain GP | mean function | verdict |
|---|---:|---:|---|
| uniformity | **-0.4688** | none | below the null → **exploration only** |
| optoelectronic | **-0.7038** | none | below the null → **exploration only** |
| thickness | **+0.5814** | **none — withdrawn 2026-09-06** | **learnable**, on its rank permutation |

**No objective carries a physics prior any more.** The thickness mean function
`log T ~ log(speed_1)+log(precur_conc)` was withdrawn on 2026-09-06: its
justification was that the fitted speed exponent agreed with theory's -0.5, and
the 95% interval on that exponent is [-0.385, -0.126], which excludes -0.5 by 4.1
standard errors. Fixing the exponents at the theoretical values scores +0.5600,
worse than no trend at all. Full table in CAMPAIGN_STATUS.md.

**Still only one learnable axis**, as on v3 — but that no longer argues for heavy
exploration. `beta = 36` was chosen on the premise that exploring wider was how the
two dead axes would come alive. The extended C1&C2 sheet (45 rows = 15 recipes made
three times) shows it is not: optoelectronic is 84.5% between-campaign drift with a
recipe ICC of **0.000**, and uniformity is reproducible (ICC **0.730**) but not
predictable from ten inputs at fifteen distinct recipes. Neither is reachable by
any beta.

**The campaign runs `beta = 4.0` and `radius = 0.25`, with no trust region** —
set on 2026-09-03 and re-confirmed before R1. At beta = 36 the radius knob was
provably inert — radii 0.15, 0.25 and 0.35 return bit-identical batches — and 18
of 50 proposed coordinates sat on a grid bound; at beta = 4 / radius 0.25 that
falls to 11 at seed 73 (11–16 across seeds, so a weak discriminator on its own).
The choice was re-checked on a box-plot sweep (beta 1–25 × radius 0.05–0.45, five
acquisition seeds) against a random-acquisition baseline, which is the honest
"did it climb?" reference; that sweep is the group's internal material and is not
in the repository. See CAMPAIGN_STATUS.md, "Are beta = 4.0 and radius = 0.25
defensible?", including why the hypervolume column of that table must not be read
as a ranking.

**Uniformity and optoelectronic are read from the workbook, not computed.** No
independent recomputation exists under this contract. The formula fingerprints
notice a changed *definition*; nothing here can notice a value that has gone
stale. That is the price of taking the score as the interface, and it is paid
deliberately.

**The v3 photoconductance inversion is fixed.** Its normalised column ranked
backwards against its own raw measurement (Spearman -0.5484, p = 0.0343); on v4
the same comparison gives **+1.0000**. Issue 10 is closed. The diagnostic stays on
because the failure is silent when it recurs.

**Thickness has no mean function — the prior was withdrawn on 2026-09-06 — and it
still earns `learnable` on its rank permutation test**, re-measured on the plain
GP after the withdrawal: observed rank ρ **+0.8036**, null mean −0.1762 (sd
0.4070), null 95th percentile +0.5607, **9 exceedances in 1800**, **p = 0.0056,
95% CI [0.0021, 0.0090]**. Removing the prior cost R² (+0.7423 → +0.5823), not the
verdict. `structured_mean.py` stays in the package, wired and tested, for a future
prior that earns its place; the live config declares none, and a test pins that.

## What is actually open

1. **R2 is next.** R1 was proposed on 2026-09-07 at seed 73 (batch hash
   `addc7edcabb0ac02`, reproduced by a second route before any film was made) and
   run in triplicate; the results came back on 2026-09-10. The filled worklist
   sits beside the workbook as `<workbook>_R1_Candidates.xlsx`, and the launcher
   proposes R2 from the R0 rows plus the five R1 condition means.
2. **Option C for R2's model** (chosen 2026-09-06): the GP gets each R1
   condition's mean plus the between-film variance measured from its triplicates
   — `model.observation_noise: replicate_pooled` — rather than the 15 raw rows,
   which carry the same information. `replicate_pooled` pools ONE variance per
   objective across conditions (a per-condition variance would rest on 2 dof);
   each row then gets that variance divided by its film count, so each R0 row, a
   single film, gets it whole.
3. **`anneal_temp` sits at a range edge in proposed conditions.** If the group
   would never anneal below some temperature, that belongs in `constraints:` —
   now a live list with three entries, so adding one is a two-line change.

## Three floors. Check all three before comparing any two numbers.

These are method, and they carry across both campaigns.

- **Null, −0.148 at N=15 — AND IT IS NOT A SIGNIFICANCE THRESHOLD.** Measured
  2026-09-04, 300 permutations with the campaign's own model: the fitted GP's
  null has median −0.4210 and 95th percentile **+0.2890**, and **28.7% of
  pure-noise shuffles beat −0.1480**. Below it a model has certainly learned
  nothing; above it means nothing on its own. Use the rank permutation test,
  or `scripts/raw_component_screen.py --calibrate` for a candidate's own bar.
  Predicting the leave-one-out mean gives
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

- **Each contract's objectives are different quantities.** A shared
  `contract_version` would make their hypervolumes look comparable when they
  measure different spaces. That is why every redefinition arrives as a new
  config file rather than an edit -- three times now.
- **Uniformity and optoelectronic are read as stored, for good** (2026-09-06).
  Do not re-derive them from their components or re-encode the group's formulas:
  the score value is the interface, and the raw-component screen (2026-09-05)
  found that every component fails on its own anyway.
- **No objective carries a prior.** The thickness mean function was withdrawn on
  2026-09-06 after its physics justification failed. Bringing one back is a
  design decision for the group, and it needs a permutation test of its own.
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
