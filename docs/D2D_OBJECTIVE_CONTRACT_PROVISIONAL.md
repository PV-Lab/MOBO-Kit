# D2D Objective Contract — Provisional, Not Approved

## Status

This document records current information without turning it into a production
decision. `configs/d2d_step2a_provisional.yaml` deliberately contains
placeholders and `approved_for_production: false`; the production validator must
reject it.

## Updated workbook profile

The updated private workbook was audited read-only from an ignored, explicitly
supplied local path. Its filename and content identity are intentionally absent
from tracked documentation.

- Sheet and content range: `Sheet1`, `A1:AC18`
- 29 columns; 15 recipe rows in Excel rows 2–16
- Canonical optimizer inputs: B:K
- Explicit display alias: `precur_vol (uL)` to `precur_vol`
- No second annealing-temperature column
- Scores: T `Uniformity score`, U `Optoelectronic score`
- Thickness: V raw average, W normalized target-score candidate
- Y has a blank header and must remain visible in the audit
- No formulas or populated objective values

The local workbook contains an off-grid input discrepancy relative to the
provisional design contract. The v2 auditor exposes the discrepancy and does
not snap or reinterpret private recipe values. The meeting must decide whether
the source is a transcription issue, an intentional off-grid execution, or
evidence that the approved input grid needs revision.

The supplied instruction text states that the row-18 notes are in P18/R18.
Direct cell inspection of the supplied workbook instead finds the note text in
**Q18/S18**, under the normalized PL and normalized photoconductance columns;
P18/R18 are blank. The contract fixture follows the pack's P18/R18 locations,
while a separate sanitized anomaly fixture and the local audit report the
Q18/S18 deviation without changing the workbook.

## Current scientific intent

The current, still provisional goals are:

1. maximize uniformity performance;
2. maximize optoelectronic performance; and
3. match thickness to 650 nm.

Aleks identified L (`Coverage`), N (`1 - Uniformity`), and O (`Phase purity`) as
uniformity components, and Q/S as normalized optoelectronic components. This is
not enough to calculate T or U: equations, weights, anchors, clipping, and
failure semantics remain unresolved.

## Unresolved objective-source choice

Step 2A supports both possible architectures but approves neither:

| Candidate mapping | Model outputs | Utility treatment |
|---|---|---|
| `T,U,W` | three precomputed scores | identity maximize only after the three columns and formulas are approved |
| `T,U,V` | two precomputed scores plus raw thickness | transform every raw thickness posterior sample through the approved target utility |

The generic engine supports a Gaussian target utility and a negative-absolute
comparison utility. The workbook's `sigma = 250` header is not treated as final
approval for either the model-source choice or the production transform.

## Objective specification requirements

Each approved objective must ultimately provide:

- a unique objective name and exact workbook/model source column;
- goal (`maximize`, `minimize`, or `target`);
- a versioned transform and fixed parameters;
- fixed affine anchors or a declaration that the source is an already-approved
  utility;
- clipping behavior;
- formula ownership and missing/failure behavior.

The transform outputs all-maximize utilities. Pareto analysis, UCB-HVI,
qLogNEHVI, and the reference point must all use that same transformed space.

## Decisions required before production

- final `T,U,W`, `T,U,V`, or other mapping;
- exact T and U formulas, or confirmation that externally approved values are
  entered directly;
- final thickness model source, target transform, sigma/scale, and symmetry;
- fixed utility scales, clipping, and fixed utility-space reference point;
- R0 row inclusion, control, replicate, QC, missing-data, failure, and outlier
  policies;
- complete physical/equipment constraints, including an explicitly approved
  empty list if none apply;
- R1 beta, posterior policy/sample count, pool size, and seed policy;
- R2 sample count, pool size, comparator policy, and seed policy;
- local-penalty radius and hard distance settings;
- approval provenance and a frozen resolved-config hash.

Until these are encoded and approved, only schema audits and explicitly
synthetic low-level examples are permitted.
