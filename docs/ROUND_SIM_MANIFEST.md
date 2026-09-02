# `manifest.csv` schema

> **This document describes TEST DATA.** The workbook and contract it reports on
> (`d2d-objectives-v2-nm-thickness`, `Summary Table.xlsx`) existed to develop and
> check the toolkit, not to run an experiment. **The real campaign is v4** --
> `configs/campaign_d2d_perovskite_final.yaml` on
> `local_inputs/Final Summary Table.xlsx`, contract `d2d-objectives-v4-final`.
> Uniformity and optoelectronic have been renormalised twice since, so **no
> number below transfers**; they describe quantities that were redefined. Start
> from `docs/CAMPAIGN_STATUS.md` for the real campaign.

Written by `scripts/plot_round_simulation.py` to
`local_outputs/round_simulations/manifest.csv`. One row per parameter cell.

This file, not the figures, is the decision instrument. The figures show what a
landscape looks like; the manifest answers whether changing a knob changed
anything, which is the question the sweep exists to settle. A knob that produces a
byte-identical batch at every setting is inert on this problem, and no amount of
looking at contour plots will tell you that.

Column order is pinned by `MANIFEST_COLUMNS` in the script and asserted by
`tests/test_plot_round_simulation.py::test_manifest_row_has_exactly_the_declared_columns`.

| column | type | meaning |
|---|---|---|
| `condition_id` | int | 1-based position in the run. Not stable across different `--conditions` filters; use the slug or `(radius, beta)` to join. |
| `arm` | str | `radius` (beta held at 4), `beta` (radius held at 0.25), `both` (the shared 0.25/4 anchor), or `grid` under `--full-grid`. |
| `radius` | float | `local_penalization.radius` for this cell. |
| `beta` | float | `rounds.r1.beta` for this cell. |
| `min_batch_distance` | float | Always 0.15. Pinned, never swept, so "spacing" means one thing in every row. |
| `seed` | int | 73 unless `--seed` overrides. The oracle, both acquisitions, and every pool draw use it. |
| `r1_batch_hash` | str | 16 hex chars. SHA-256 of the R1 conditions, **sorted** and rounded to 12 dp. Two cells sharing a hash proposed the same set of recipes; order is not part of the identity. |
| `r2_batch_hash` | str | The same for R2. |
| `r1_min_pairwise_distance` | float | Smallest normalised distance within the R1 batch. Compare against `radius` to see whether local penalisation had anything to act on. |
| `r2_min_pairwise_distance` | float | The same for R2. |
| `r1_boundary_coords_total` | int | How many coordinates across the whole R1 batch sit exactly at a range edge. On the live campaign this is driven by the monotone `anneal_temp` mean function, not by either knob. |
| `r2_boundary_coords_total` | int | The same for R2. |
| `r1_boundary_coords_per_condition` | JSON list | Per-condition breakdown, so one pinned condition is distinguishable from five mildly-pinned ones. |
| `r2_boundary_coords_per_condition` | JSON list | The same for R2. |
| `hv_r0` | float | Hypervolume of the 15 oracle-scored R0 points, utility space, at the campaign's declared `reference_point_utility`. |
| `hv_r0_r1` | float | After adding the 5 R1 conditions. |
| `hv_r0_r1_r2` | float | After adding the 3 R2 conditions. |
| `hv_gain_r1` | float | `hv_r0_r1 - hv_r0`. |
| `hv_gain_r2` | float | `hv_r0_r1_r2 - hv_r0_r1`. |
| `baseline_hv_reported_by_r1` | float | The observed HVI baseline the R1 acquisition actually used, from `run_r1_ucb`'s own diagnostics. |
| `baseline_hv_independent` | float | The same quantity recomputed by the script through `metrics.compute_ref_pareto_hv` — a different Pareto filter and a different call path. **`run_cell` raises if these two disagree.** |
| `baseline_hv_pareto_size` | int | How many observations sit on the baseline Pareto front. Under the historical mis-encoding this was 2; correctly encoded it is 5. |
| `baseline_hv_unencoded_contrast` | float | What the baseline *would* be if measurement-space values reached the transform directly — the size of the defect fixed in commit `4b76670`. Constant across rows, and **never expected to equal anything**. |
| `r1_fit_warnings` | int | Fit-guard warnings raised by the GP that proposed R1. Guard warnings only, not the ~18 numpy deprecation notices per fit. |
| `r2_fit_warnings` | int | The same for the R2 model. |
| `final_fit_warnings` | int | The same for the 23-point model the heatmaps render. Non-zero puts a banner on that cell's figures. |
| `mean_utility_r0` | float | Mean utility over all objectives and all 15 R0 points. A summary, not a ranking: it averages three objectives that are not commensurable. |
| `mean_utility_r1` | float | The same over the 5 R1 conditions. |
| `mean_utility_r2` | float | The same over the 3 R2 conditions. |

## Reading it

**Hypervolume rises monotonically by construction.** Adding points can only grow a
Pareto front, so `hv_r0 <= hv_r0_r1 <= hv_r0_r1_r2` holds in every row and proves
nothing on its own — random sampling satisfies it too. What carries information is
the *size* of `hv_gain_r1` compared across cells, and the batch hashes.

**Cells with equal `(r1_batch_hash, r2_batch_hash)` are the same experiment.**
Their hypervolumes and utilities are then identical by construction, not by
agreement, and quoting them as independent replicates would be double counting.

**The baseline columns are the standing tripwire for the encoding defect.**
`reported` comes from inside the acquisition; `independent` is recomputed by a
different route; `run_cell` raises rather than writing a manifest if they differ.
`unencoded_contrast` is the size of the historical mistake and is deliberately not
compared to anything — asserting all three equal would be an assertion that can
only ever fail, because the third column exists precisely to reproduce the wrong
answer. They are constant within a run, and recorded per row so a single row is
self-describing when pasted somewhere else.
