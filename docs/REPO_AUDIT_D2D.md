# MOBO-Kit Repository Audit for the D2D Campaign

## Snapshot

- Repository: `PV-Lab/MOBO-Kit`
- Verified default branch: `main`
- Baseline/main commit: `459afc6d06f66f4d5fb48c3f563e65c61f5df16f`
- Reference branch: `Nicky's-MOBO-Testground`
- Reference tip: `546a5bbdf8fc2f6f9fc6764ada5f5d587e1faa6f`
- Step 1 feature branch: `feature/d2d-mobo-step1-baseline`
- Reference comparison: 10 commits ahead, 0 behind, 59 changed paths,
  15,887 insertions, and 166 deletions

The remote was fetched before branching. The reference branch was inspected by
explicit Git refs only and was not merged.

## Architecture and current workflow

The package separates design construction (`design.py`), normalization and grid
snapping (`data.py`), constraints (`constraints.py`), LHS (`lhs.py`), independent
GP fitting (`models.py`), qNEHVI/qLogNEHVI (`acquisition.py`), metrics
(`metrics.py`), plots (`plotting.py`), the high-level runner (`main.py`), and CLI
dispatch (`cli.py`). That is a workable engine boundary for a future thin Excel
adapter.

The existing runner assumes all objectives are maximized, fits one `SingleTaskGP`
per output in a `ModelListGP`, continuously optimizes qNEHVI/qLogNEHVI in the
normalized cube, snaps to the physical grid, and writes an append-style CSV. It
has no campaign/round/sample/status state model and neither branch contains the
required R1 UCB.

## Verified main-branch defects

1. **Broken YAML-to-LHS path.** `generate_initial_experiments` passed a raw config
   dictionary to `build_design`, which expects `InputSpec` objects.
2. **First experiment dropped.** `pandas.read_csv` had already consumed the
   header, but `split_XY` sliced at `iloc[6:]`; the first demo experiment is at
   DataFrame row 5.
3. **Fixed-offset parsing.** CSV metadata, separator, and data rows were assumed
   rather than detected. Plain data and malformed metadata were accidental
   behaviors.
4. **Implicit constraint.** Generic CSV conversion always injected the legacy
   humidity/temperature Clausius-Clapeyron constraint.
5. **Unsafe metadata defaults.** Malformed start/stop/step cells could be replaced
   with generic numeric defaults instead of failing.
6. **Inconsistent public types.** `split_XY` was annotated to return arrays but
   returned DataFrames; `csv_to_config` was annotated as `str` but returned a
   dictionary and wrote YAML as a side effect.
7. **Incomplete objectives accepted.** Partial/all-blank objective rows could
   reach tensor/GP code as NaNs; blanks were not distinguished from real zeros.
8. **Non-deterministic/unsafe LHS.** Attempts were reseeded, subset selection used
   a separate unseeded RNG, snapped duplicates were not removed, correlation
   thresholds could be violated, and the final fallback bypassed constraints.
9. **Latent plotting failure.** LHS diagnostics referenced nonexistent
   `design.labels`.
10. **Reference-point mismatch.** The high-level runner computed one reference
    for reporting, then optimized with a hard-coded `[-0.01] * M` vector.
11. **Candidate failure hidden.** Proposal exceptions and short/empty batches
    could still lead to a top-level `status: success`.
12. **Grid collisions/observed duplicates.** Continuous optima can snap to the
    same recipe; the main path has no final deduplication or observed-point
    exclusion.
13. **Round-trip mismatch.** LHS emits input-only CSV while the runner expected a
    metadata-style CSV with objectives.
14. **CLI drift.** README examples omitted the required `run` subcommand and
    `--num-restarts` was parsed but unused.
15. **Packaging drift.** Install URLs and the `all` extra still referenced
    MOBO-FOM; dependency minimums admitted mutually incompatible future stacks.
16. **Stale tests.** Existing tests imported removed `src.*` modules and expected
    APIs/files that no longer exist, so they did not validate the packaged code.
17. **Tracked caches.** Main tracked Python bytecode under `src/__pycache__` and
    `tests/__pycache__` despite ignore rules.

The main branch also contains pre-existing demonstration result images/CSVs.
Step 1 classifies them as published demo artifacts and does not regenerate or
expand them; generated campaign outputs and private inputs remain ignored.

## D2D reference-branch findings

The reference branch contains useful research ideas: large discrete candidate
pools, observed-point exclusion, deduplication, discrete qNEHVI/local search,
normalized linear constraints, posterior diagnostics, conservative noise/kernel
options, and encoding fallback. These need selective review and synthetic tests
in later steps.

It is not an authoritative executable D2D pipeline:

- its D2D YAML defines ten inputs but enables a Clausius-Clapeyron constraint on
  `absolute_humidity` and `temperature_c`, neither of which exists in that design;
- its D2D CSV is inherited from the eight-input slot-die example, conflicts with
  the YAML, and has declared/data field-count mismatches;
- notebooks import `MixedMCMultiOutputObjective`, which is absent from the active
  package and exists only in a stale `src/mobo_kit OLD` copy;
- notebook objective directions include a 650-nm thickness match, but qNEHVI
  calls omit that transform while using a transformed reference point;
- posterior non-domination calculations then compare raw outputs as if all were
  maximized;
- the automatically derived reference changes with the observed dataset, so
  round-to-round hypervolume would not be comparable;
- the diversity selector is post-hoc Euclidean reranking, not documented local
  hypervolume penalization, and its fallback can bypass the configured minimum
  distance;
- notebooks contain personal macOS Dropbox paths, undefined variables,
  inconsistent `str`/`Path` operations, stale campaign names, embedded outputs,
  and saved execution errors;
- caches, egg-info, a complete old package copy, and generated results are
  committed; and
- no tests accompany the large acquisition/model changes.

The changed `results/experiment/next_batch.csv` is legacy slot-die data, not a
D2D R1 result. No Excel integration, stable row identifiers, or round-state
adapter exists on the branch.

## Workbook audit

An ignored private workbook was inspected read-only and identity-checked against
the explicitly supplied source. Its filename and digest are not tracked.

- one sheet: `Sheet1`
- non-empty range: `A1:AC16`
- 29 columns and 15 sample rows
- `Uniformity score` duplicates at 1-based positions 17 and 20
- related/ambiguous `anneal_temp` and `Anneal Temp` at positions 8 and 12
- blank header at position 25, after `Total combination - addition`
- measurements and derived scores are blank
- formatting extends beyond the non-empty range, so auditors must calculate the
  range from cell content rather than styled dimensions alone

No workbook cells, formulas, formats, or macros were changed.

## Step 1 remediation boundary

Step 1 fixes schema/execution ambiguity, parsing, opt-in constraints, design
construction, deterministic LHS, workbook auditing, tests, packaging, and
hygiene. It adds a canonical input-only D2D YAML with empty objective names and
no constraints.

Step 1 deliberately does not implement UCB, local penalization, qNEHVI round
changes, objective formulas, a thickness transform, a campaign reference point,
real candidates, or an Excel button. The existing qNEHVI runner is made opt-in
and requires an explicit same-space reference point so it cannot silently invent
one.
