# D2D MOBO Meeting Decision Record

**Meeting date:**
**Participants:**
**Recorded by:**
**Campaign/composition:**
**Decision-record version:**

Complete this document after the experimental-team meeting. Replace every
`PENDING` entry and record who approved it. This document should become the
source for the frozen Step 2B production configuration.

## 1. R0 dataset and row roles

- Number of R0 recipe executions: `PENDING`
- Are the 15 workbook rows the complete R0 set? `PENDING`
- Control recipe(s) and sample ID(s): `PENDING`
- Do controls train the GP? `PENDING`
- Replicate rows/groups: `PENDING`
- Replicate aggregation/noise policy: `PENDING`
- Required status/QC fields before model fitting: `PENDING`
- Missing-one-objective rule: `PENDING`
- Failed-film rule: `PENDING`
- Outlier/exclusion approval owner: `PENDING`

## 2. Final three model outputs and BO utilities

### Objective 1 — Uniformity

- Model source column: `PENDING`
- Utility/source column: `PENDING`
- Direction: `maximize`
- Is T supplied externally or calculated by Python? `PENDING`
- Exact equation using L/N/O: `PENDING`
- Component weights: `PENDING`
- Component normalization/clipping: `PENDING`
- Missing/failure semantics: `PENDING`
- Approved by: `PENDING`

### Objective 2 — Optoelectronic

- Model source column: `PENDING`
- Utility/source column: `PENDING`
- Direction: `maximize`
- Is U supplied externally or calculated by Python? `PENDING`
- Raw PL normalization anchor/formula: `PENDING`
- Raw photoconductance normalization anchor/formula: `PENDING`
- Exact Q/S combination and weights: `PENDING`
- Linear/log treatment: `PENDING`
- Missing/zero/failure semantics: `PENDING`
- Approved by: `PENDING`

### Objective 3 — Thickness

- Model source: `V raw thickness` / `W precomputed score` / other: `PENDING`
- Target: `650 nm` — confirm: `PENDING`
- Utility transform: `PENDING`
- Sigma/tolerance/scale: `PENDING`
- Symmetric about target? `PENDING`
- Clipping/bounds: `PENDING`
- Is W calculated by Python or supplied externally? `PENDING`
- Approved by: `PENDING`

### Final objective-column decision

- Final mapping: `T,U,W` / `T,U,V` / other: `PENDING`
- Rationale: `PENDING`

## 3. Fixed utility scales and reference point

- Are all final BO utilities on `[0,1]`? `PENDING`
- Fixed scale/anchor for Uniformity: `PENDING`
- Fixed scale/anchor for Optoelectronic: `PENDING`
- Fixed scale/anchor for Thickness utility: `PENDING`
- Out-of-range/clipping policy: `PENDING`
- Fixed utility-space hypervolume reference point: `PENDING`
- Reference-point rationale: `PENDING`
- Must remain fixed across R0/R1/R2? `PENDING`
- Approved by: `PENDING`

## 4. Process and equipment constraints

List every rule in physical units. Use `NONE — constraints: [] approved` only if
no rules apply.

| ID | Rule | Variables | Hard/soft | Rationale | Approved by |
|---|---|---|---|---|---|
| C1 | PENDING | | | | |

Specific checks:

- Rule when `speed_2 = 0`: `PENDING`
- Antisolvent-time relation to spin time: `PENDING`
- Allowed antisolvent volume/time combinations: `PENDING`
- Anneal temperature/time restrictions: `PENDING`
- Concentration/volume restrictions: `PENDING`
- Equipment-resolution restrictions beyond configured steps: `PENDING`
- Resolution of the local off-grid input discrepancy versus the approved input
  contract: `PENDING`

## 5. R1 UCB-HVI settings

- Method approved: `ucb_hvi` — `PENDING`
- Batch size: `5` — `PENDING`
- Beta: `PENDING`
- Equivalent kappa `sqrt(beta)`: `PENDING`
- Latent posterior or observation-noise posterior: `PENDING`
- Posterior MC samples for utility moments: `PENDING`
- Candidate pool size: `PENDING`
- Seed policy: `PENDING`
- Approval/delegation owner: `PENDING`

## 6. R2 qLogNEHVI settings

- Method approved: `qlognehvi` — `PENDING`
- Batch size: `3` — `PENDING`
- MC samples: `PENDING`
- Candidate pool size: `PENDING`
- Sequential pending-point selection approved? `PENDING`
- Native joint-q comparator required? `PENDING`
- Seed policy: `PENDING`
- Approval/delegation owner: `PENDING`

## 7. Local penalization and diversity

- Distance metric: normalized Euclidean / weighted / other: `PENDING`
- Soft penalty formula approved: `PENDING`
- Penalty radius: `PENDING`
- Minimum within-batch distance: `PENDING`
- Minimum distance from observed/pending points: `PENDING`
- May hard distances ever be relaxed? Recommended `No`: `PENDING`
- Required numerical diversity report: `PENDING`
- Required plots: `PENDING`
- Approval/delegation owner: `PENDING`

## 8. Workbook and score ownership

- Who enters raw characterization values? `PENDING`
- Who approves/enters final T/U/W scores? `PENDING`
- Should Python calculate any scores? `PENDING`
- Source-of-truth workbook location: `PENDING`
- One workbook for full campaign or one per round? `PENDING`
- Sample-ID convention: `PENDING`
- Backup/audit requirement: `PENDING`

## 9. Production approval

- Configuration version approved for dry-run: `PENDING`
- Configuration version approved for real R1: `PENDING`
- Scientific approver(s): `PENDING`
- Computational approver(s): `PENDING`
- Experimental operator approver(s): `PENDING`
- Approval date/time: `PENDING`
- Notes/conditions: `PENDING`
