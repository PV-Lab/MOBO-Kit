# D2D MOBO Campaign Data Contract (Step 1 Baseline)

## Status and scope

This document defines the machine-readable baseline for the D2D campaign. It
does **not** approve objective formulas, a thickness utility, a hypervolume
reference point, process constraints, UCB, local penalization, or production
candidate generation. Every field marked **TBD - experimental-team decision**
must be resolved before a real R1 run.

The optimization engine remains a tested Python API/CLI. A future Excel button
will be a thin adapter that validates and passes campaign data to that engine;
scientific logic must not be implemented in VBA or worksheet formulas alone.

## Canonical input grid

Input columns are ordered exactly as shown. Units belong in schema metadata,
not in numeric cells.

| Position | Name | Unit | Start | Stop | Step | Grid values |
|---:|---|---|---:|---:|---:|---:|
| 1 | `speed_1` | rpm | 1000 | 6000 | 500 | 11 |
| 2 | `time_1` | s | 5 | 50 | 5 | 10 |
| 3 | `speed_2` | rpm | 0 | 5000 | 500 | 11 |
| 4 | `time_2` | s | 10 | 60 | 5 | 11 |
| 5 | `precur_conc` | M | 1.00 | 2.00 | 0.05 | 21 |
| 6 | `precur_vol` | uL | 40 | 200 | 10 | 17 |
| 7 | `anneal_temp` | C | 100 | 185 | 5 | 18 |
| 8 | `anneal_time` | min | 10 | 60 | 5 | 11 |
| 9 | `anti_vol` | uL | 100 | 200 | 5 | 21 |
| 10 | `anti_time` | s | 9 | 25 | 2 | 9 |

The full Cartesian product contains 177,816,994,740 recipes and must never be
materialized. Candidate algorithms must use sampling, discrete pools, or
discrete local search.

For input `j`, a value `x` is on-grid only when all three conditions hold:

1. `start_j <= x <= stop_j`;
2. `(x - start_j) / step_j` is an integer within the configured numeric
   tolerance;
3. the serialized value round-trips without changing its grid index.

Duplicate input names, non-finite bounds, non-positive steps, reversed bounds,
and endpoints that are not reachable by an integral number of steps are schema
errors.

## Canonical record roles

A future campaign table should contain one row per physical recipe execution.
The following identifiers are proposed now so later round trips are unambiguous:

| Field | Role | Step 1 rule |
|---|---|---|
| `campaign_id` | Stable campaign identifier | Required later; value TBD |
| `sample_id` | Unique physical sample identifier | Required and never reused |
| `round` | `R0`, `R1`, or `R2` | Required later |
| `row_role` | `candidate`, `control`, or `replicate` | Required later; policy TBD |
| `candidate_status` | `proposed`, `run`, `measured`, `excluded`, `failed` | Required later |
| `replicate_group` | Links repeated recipes | Nullable; aggregation policy TBD |
| `include_in_model` | Explicit QC gate | Required before GP fitting; policy TBD |
| `exclusion_reason` | Human-readable audit reason | Required when excluded |

Columns then appear in these logical groups:

1. identifiers and workflow state;
2. the ten canonical optimizer inputs in the order above;
3. measured process covariates (for example, a measured rather than setpoint
   temperature);
4. raw characterization measurements;
5. transparent derived scores;
6. three approved BO utilities in transformed/model space;
7. provenance and QC fields.

Raw measurements must never be overwritten by cleaning or derived scores.
Blank measurements mean missing/unmeasured, not zero. Parsing preserves missing
values; the model boundary rejects partial or all-blank objective rows rather
than silently filling or dropping them.

## Workbook findings and mapping guardrails

The supplied private workbook was audited read-only. Header positions below are
1-based Excel column positions.

- Sheet: `Sheet1`
- Non-empty data range: `A1:AC16`
- Columns: 29
- Sample rows: 15
- Canonical inputs: columns B:K
- `anneal_temp`: column H
- related but distinct `Anneal Temp`: column L
- duplicate `Uniformity score`: columns Q (17) and T (20)
- blank header: column Y (25), immediately after
  `Total combination - addition`
- characterization and score cells are blank in the supplied copy

The two anneal columns must not be merged automatically. Their meanings
(setpoint, measured temperature, second anneal, or legacy field) are **TBD -
experimental-team decision**. The two `Uniformity score` columns must remain
position-addressable until both meanings are approved. A reader must inspect raw
header cells before any library can rename duplicates.

## Measurement, score, and objective roles

The workbook currently exposes raw or semi-processed fields such as `Coverage`,
`Uniformity`, `Phase purity`, `PL - Implied Voc (Max)`, `Photoconductance (Max)`,
and `Thickness (avg)`, plus several derived scores. None is automatically a BO
objective.

- Objective 1 source/formula/direction: **TBD - experimental-team decision**
- Objective 2 source/formula/direction: **TBD - experimental-team decision**
- Objective 3 source/formula/direction: **TBD - experimental-team decision**
- Thickness target, tolerance, and transform: **TBD - experimental-team
  decision**
- Fixed scaling limits and clipping policy: **TBD - experimental-team decision**
- Fixed campaign reference point in original and transformed units: **TBD -
  experimental-team decision**

All three approved utilities and the reference point must be transformed by the
same versioned contract. Round-by-round min/max scaling must not be introduced
without explicit approval because it would make hypervolume incomparable.

## Campaign state machine

```text
CONFIGURED
  -> R0_PROPOSED (deterministic, grid-valid LHS; count TBD)
  -> R0_RUNNING
  -> R0_MEASURED
  -> R0_VALIDATED (QC/objective mapping/reference frozen)
  -> R1_PROPOSED (5 candidates; approved multi-objective UCB TBD)
  -> R1_RUNNING
  -> R1_MEASURED
  -> R1_VALIDATED
  -> R2_PROPOSED (3 candidates; approved qNEHVI + penalization policy TBD)
  -> R2_RUNNING
  -> R2_MEASURED
  -> COMPLETE
```

Transitions fail closed. Missing required measurements, duplicate sample IDs,
unapproved configuration fields, hard-constraint violations, an undersized
candidate batch, or a code/config hash mismatch prevent transition to the next
proposal state.

## Proposed future workbook sheets

This is an interface proposal, not a Step 1 workbook edit.

- `Campaign`: locked campaign identity, code commit, schema version, approved
  objective contract, fixed reference point, and round settings.
- `Experiments`: flat editable rows containing identifiers, inputs, raw
  measurements, QC, and derived utilities.
- `Next Candidates`: generated worklist with candidate IDs, physical conditions,
  acquisition diagnostics, and validation state.
- `Diagnostics`: Pareto/hypervolume history, model checks, pairwise candidate
  distances, and acquisition plots.
- `Decision Log`: approvals, changes, warnings, operator, timestamps, input-file
  hash, resolved configuration hash, seed, environment versions, and Git commit.

## Validation and reproducibility requirements

- Canonical names and order are exact; unknown input columns are errors unless
  explicitly registered as measured covariates.
- Every proposed input is finite, within bounds, and exactly on-grid.
- A proposed batch has the requested number of unique recipes or fails clearly.
- Constraints are opt-in, evaluated in physical units after snapping, and never
  relaxed silently.
- Controls and replicates are never inferred from identical recipes; their row
  roles are explicit.
- Model-boundary inputs/objectives are numeric and complete according to the
  approved QC policy.
- Every production run records: campaign/schema version, resolved config, source
  workbook hash, row selection/QC decisions, random seed, dependency versions,
  Git commit, acquisition name/parameters, reference point, candidates, and
  diagnostics.

## Unresolved decisions blocking real R1

1. R0 experimental count, control count, and whether/how controls train the GP.
2. The three authoritative objective formulas and source columns.
3. Uniformity definition, direction, coverage/phase-purity roles, and duplicate
   score meanings.
4. Thickness target, tolerance, and utility transform.
5. Optoelectronic score formula and failure/zero semantics.
6. Fixed objective scales and out-of-range handling.
7. Fixed campaign hypervolume reference point.
8. Meanings and roles of `anneal_temp` and `Anneal Temp`.
9. Explicit process/equipment/safety constraints in physical units.
10. Literature-backed multi-objective UCB definition and beta policy.
11. Local-penalization metric, radius/weight, observed-point treatment, and hard
    minimum-distance policy.
12. Missing-data, failed-film, QC, outlier, and replicate aggregation policy.
13. Excel deployment environment, macro policy, Python installation, and code
    signing/trusted-location requirements.

Until these decisions are approved, the D2D YAML contains no objective names and
no constraints, and the public runner requires explicit opt-in plus an explicit
reference point before invoking its existing qNEHVI proposal path.
