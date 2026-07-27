# Step 1 Handoff

## 1. Status

- Historical implementation branch: `feature/d2d-mobo-step1-baseline`
- Base commit: `459afc6d06f66f4d5fb48c3f563e65c61f5df16f` (`origin/main`)
- Reference branch inspected only: `Nicky's-MOBO-Testground` at
  `546a5bbdf8fc2f6f9fc6764ada5f5d587e1faa6f`
- Original milestone state: Step 1 was reviewed locally before its checkpoint
  commit. This handoff preserves that historical state; publication later uses a
  clean squash containing the sanitized Step 1 through Step 2C tree.
- Overall result: **PASS**, with the documented deviations in section 11

No production R1/R2 candidates were generated. No UCB implementation, local
penalization, final objective transform, thickness target, or campaign reference
point was introduced.

## 2. Summary of changes

- Defined the exact ten-input D2D grid and campaign boundary without inventing
  objectives or constraints.
- Replaced fixed-row CSV handling with a validated metadata-style parser that
  preserves the first experiment and rejects ambiguous/incomplete model data.
- Corrected config-to-design construction and made grid validation explicit.
- Made constraints opt-in and removed the invalid D2D humidity constraint.
- Reworked R0 LHS generation to be deterministic, exactly sized, grid-valid,
  unique after snapping, post-snap constrained, and fail-closed.
- Added a read-only raw-cell workbook schema auditor and a sanitized fixture.
- Pinned a CPU-tested dependency stack and repaired package/CLI documentation.
- Made candidate generation opt-in, require an explicit reference point, and
  reject incomplete, duplicate, off-grid, non-finite, or observed recipes.
- Moved default generated output to ignored `local_outputs/`, enabled headless
  plot generation, and made explicit missing config paths fail.
- Removed tracked bytecode caches and retired three obsolete experimental test
  scripts whose active behavior is covered by the replacement test suite.

## 3. Files changed

| File or group | Purpose |
|---|---|
| `.gitignore` | Ignore private inputs, local outputs, caches, environments, coverage, and build artifacts. |
| `README.md` | Correct install, package, CLI, CPU/GPU, metadata-CSV, output, and round-trip guidance. |
| `pyproject.toml`, `setup.py`, `requirements.txt` | Normalize packaging, supported Python range, extras, dependency bounds, and pytest collection. |
| `requirements/constraints.txt`, `requirements/dev.txt` | Pin the tested direct stack and provide a reproducible editable development install. |
| `configs/FA0.9CS0.1PbI3_260407_Config.yaml` | Canonical ten-input D2D grid, empty objectives, and `constraints: []`. |
| `docs/D2D_CAMPAIGN_SPEC.md` | Canonical schema, workbook mapping guardrails, state model, and unresolved decisions. |
| `docs/REPO_AUDIT_D2D.md` | Verified main/reference-branch architecture, defects, risks, and workbook findings. |
| `docs/STEP1_HANDOFF.md` | This implementation and verification record. |
| `src/mobo_kit/design.py` | Strict `InputSpec`, endpoint-aligned grids, and supported config-to-design path. |
| `src/mobo_kit/utils.py` | Raw CSV metadata parser, explicit encodings, typed DataFrames, and strict model-boundary validation. |
| `src/mobo_kit/constraints.py` | Explicit-only constraint parsing and clear configuration/shape errors. |
| `src/mobo_kit/lhs.py` | Deterministic snapped-grid LHS with uniqueness, exact-size, constraint, and hard-correlation guarantees. |
| `src/mobo_kit/workbook_schema.py` | Read-only, content-range workbook audit preserving raw duplicate/blank headers. |
| `src/mobo_kit/main.py`, `src/mobo_kit/cli.py` | Correct design construction, safe defaults, CLI wiring, headless output, and proposal guards. |
| `tests/test_csv_parser.py` | Metadata boundary, first-row, encoding, objective, malformed-data, and return-type tests. |
| `tests/test_design.py`, `tests/test_lhs.py` | Grid/schema validation and deterministic/unique/constrained LHS regressions. |
| `tests/test_d2d_baseline.py` | Exact D2D contract/cardinality, 20-row smoke, constraints, imports, CLI, hygiene, and fail-closed guards. |
| `tests/test_workbook_schema.py` | Sanitized workbook fixture plus optional ignored-local-workbook audit. |
| `tests/test_acquisition.py`, `tests/test_models.py`, `tests/test_plotting.py` | Active-package, CPU-fast replacements for stale tests. |
| tracked `src/__pycache__/*`, `tests/__pycache__/*` | Removed 26 committed bytecode/cache artifacts. |
| `tests/smoke_test.py`, `tests/simple_gp_test.py`, `tests/synthetic_test.py` | Retired obsolete scripts using missing `src.*` APIs and uncontrolled experimental/candidate loops. |

The private workbook remains only in ignored local storage. Its runtime identity
matches the explicitly supplied source, but neither filename nor digest is
tracked.

## 4. Defects fixed

1. `generate_initial_experiments` passed a raw dict to `build_design`.
2. CSV parsing used a fixed row offset and dropped the first experiment.
3. Duplicate headers could be mangled before ambiguity was reported.
4. Malformed metadata, missing/partial objectives, and nonnumeric model rows
   could pass too far or be handled inconsistently.
5. Generic config conversion injected an unrelated Clausius-Clapeyron
   constraint; the D2D config referenced nonexistent variables.
6. Public data-loading annotations and runtime return types disagreed.
7. LHS retries/subset choice were not fully deterministic and could return
   duplicates, constraint violations, an undersized set, or a correlation-limit
   violation.
8. LHS diagnostics referenced nonexistent `design.labels`.
9. The runner used an internally inconsistent hard-coded proposal reference
   point, hid proposal failure, and did not wire `--num-restarts` through.
10. Proposal output could report success with an incomplete, duplicate,
    off-grid, or already observed batch.
11. An explicitly supplied missing config path silently triggered schema
    inference in the Python API.
12. Plot generation depended on Tcl/Tk even though the runner only writes files.
13. Default commands could overwrite tracked demonstration results.
14. README/package names, URLs, extras, CLI examples, and dependency policy had
    drifted from MOBO-Kit.
15. Tracked caches and stale `src.*` test scripts obscured the actual package
    test surface.

## 5. Tests added

1. Valid and invalid YAML/config to `Design`, including exact D2D input order,
   bounds, steps, per-axis grids, and full product `177,816,994,740`.
2. Robust metadata CSV parsing: optional separator, first data row, encodings,
   duplicate headers, missing objectives, malformed metadata, plain CSV, and
   empty experimental data.
3. Model-boundary DataFrame behavior for missing, blank, partial, nonnumeric,
   and non-finite values.
4. Empty/default, valid explicit, and missing-column constraint behavior.
5. Same-seed determinism, different-seed change, grid membership, bounds,
   uniqueness, post-snap constraints, exact size, impossible requests, and hard
   correlation limits for LHS.
6. A deterministic unique 20-by-10 D2D R0 smoke generation; the sample count is
   still caller-configurable.
7. Sanitized and optional local workbook audits for range, sample count,
   duplicate headers, blank header, anneal ambiguity, and file immutability.
8. Direct Torch/GPyTorch/BoTorch/package CPU imports and all three CLI help
   surfaces.
9. Production source/notebook personal-path scan.
10. Fail-closed proposal batch validation and explicit missing-config behavior.
11. Active-package acquisition, model, and headless plotting smoke tests.

## 6. Commands run

```powershell
git -c http.sslBackend=openssl fetch --all --prune
git switch -c feature/d2d-mobo-step1-baseline main

# Clean environment installation used the committed constraints.
uv pip install -c requirements/constraints.txt -e ".[dev]"

.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp <writable external temp>

.\.venv\Scripts\python.exe -m black --check <all changed/new Python files>
git diff --check
git ls-files

.\.venv\Scripts\mobo-kit.exe --help
.\.venv\Scripts\mobo-kit.exe generate --help
.\.venv\Scripts\mobo-kit.exe run --help
.\.venv\Scripts\mobo-kit.exe run `
  --csv data/processed/configCSV_example.csv `
  --config configs/demo_config.yaml --device cpu --seed 42 --verbose `
  --out <writable temporary output>

.\.venv\Scripts\python.exe -c "import torch, gpytorch, botorch, mobo_kit"
.\.venv\Scripts\python.exe -c "from mobo_kit.workbook_schema import audit_campaign_workbook; ..."
```

GitHub access, branch tips, and comparison were also verified against the
connected repository before implementation. The reference branch was never
checked out over or merged into the feature branch.

## 7. Test results

Final complete test command (after all code changes):

```text
79 passed, 14 third-party Matplotlib/PyParsing deprecation warnings,
0 skipped, 0 failed
```

The literal `pytest -q` command also passes. This Windows sandbox denies pytest's
default cache directory, producing one additional cache warning; the recorded
clean result disables only the cache provider and uses an explicit writable
temporary base. `git diff --check`, `pip check`, CLI help, direct imports, the
safe CPU demo run, workbook hash/mtime checks, and the scoped Black check pass.

The safe demo run loaded 12 rows with 8 inputs and 3 synthetic/demo objectives,
fit the CPU models, wrote `parity_plots.png`, wrote no `next_batch.csv`, and
exited 0.

## 8. Tested environment

- OS: Microsoft Windows NT `10.0.26200`
- Python: `3.12.10`
- torch: `2.8.0+cpu`
- gpytorch: `1.14`
- botorch: `0.15.1`
- linear_operator: `0.6`
- numpy: `2.2.6`
- scipy: `1.16.0`
- pandas: `2.3.1`
- scikit-learn: `1.7.1`
- matplotlib: `3.10.3`
- seaborn: `0.13.2`
- PyYAML: `6.0.2`
- Excel reader: openpyxl `3.1.5`
- Image handling: Pillow `12.3.0`
- pytest: `8.4.1`
- black: `25.1.0`
- CUDA available: `False`

The committed package range is Python `>=3.11,<3.13`. Python 3.10 was removed
from the advertised baseline because the tested SciPy 1.16 stack requires
Python 3.11 or newer.

## 9. Workbook audit

- Input: explicitly supplied ignored private workbook
- Identity: verified locally before and after the audit; not tracked
- Sheet: `Sheet1`
- Used/non-empty range: `A1:AC16`
- Columns: 29
- Sample rows: 15
- Duplicate headers: `Uniformity score` at one-based columns 17 (`Q`) and 20
  (`T`)
- Ambiguous headers: `anneal_temp` at column 8 (`H`) and `Anneal Temp` at
  column 12 (`L`)
- Other warning: blank header at column 25 (`Y`), immediately after
  `Total combination - addition`
- Modification result: SHA-256 and modification time unchanged; no save was
  performed

No duplicate or related workbook headers were merged, renamed, or assigned a
scientific meaning.

## 10. Unresolved scientific decisions

1. R0 candidate/control counts and whether controls/replicates train the GP.
2. The three authoritative BO objective columns, formulas, and directions.
3. Uniformity definition and meanings of both `Uniformity score` columns.
4. Thickness target, tolerance, and utility transformation.
5. Optoelectronic score formula and missing/failed/zero semantics.
6. Fixed campaign objective scaling and out-of-range policy.
7. Fixed hypervolume reference point in original and transformed units.
8. Meanings of `anneal_temp` and `Anneal Temp` and which is the optimizer input.
9. Physical/equipment/safety constraints and control recipes.
10. Literature-backed multi-objective UCB definition and beta/noise policy.
11. Local-penalization metric, radius/weight, observed-point treatment, and hard
    minimum distance.
12. Missing-data, failed-film, QC, outlier, and replicate aggregation policy.
13. Excel deployment environment, macro policy, Python installation, signing,
    and trusted-location requirements.

## 11. Deviations from specification

- The pack preferred Python 3.10/3.11. The usable managed local runtime was
  Python 3.12.10, so that exact CPU environment was tested. The resulting
  package supports 3.11-3.12, but Python 3.11 still needs CI verification.
- Native pip dependency resolution repeatedly consumed CPU without terminating
  cleanly in this managed environment. The environment was installed with `uv`
  against the committed pip-compatible constraints; `pip check` and a local
  no-dependency editable-install dry run pass.
- All changed/new Python files pass Black. A repository-wide Black check still
  reports five untouched legacy source modules (`acquisition.py`, `data.py`,
  `metrics.py`, `models.py`, and `plotting.py`) as style-only reformat targets.
  They were not mass-formatted to avoid an unrelated full-source rewrite.
- The real-workbook integration test ran locally. It remains optional/skipped
  for clean clones where the ignored private workbook is unavailable.

## 12. Risks and limitations

- The D2D YAML deliberately has no objective names; it is valid for R0 design
  generation but cannot authorize a D2D model/acquisition run.
- R0 `generate` emits an input-only CSV. It does not round-trip directly into
  the metadata-style `run` parser; the future workbook adapter must implement
  that state transition.
- The existing qNEHVI path remains a legacy, explicit opt-in capability. Its
  output is guarded for size/grid/uniqueness/observed recipes, but Step 1 does
  not add round logic or local penalization.
- Only CPU/Python 3.12 was executed locally. GPU and Python 3.11 need separate
  CI coverage before being claimed as tested campaign environments.
- The 14 test warnings are third-party Matplotlib/PyParsing deprecations.

## 13. Recommended Step 2

- Obtain and encode the approved objective-transform contract, fixed scales,
  QC policy, process constraints, and fixed reference point.
- Implement and mathematically document the approved R1 multi-objective UCB
  acquisition using synthetic tests first.
- Add one reusable, fail-closed local-penalized batch-selection policy for R1
  and R2, including distance diagnostics and exact batch-size guarantees.
- Design the workbook round-trip/state adapter only after the scientific schema
  is approved; keep optimization logic in the Python engine.
- Do not generate real R1 candidates until those approvals and a frozen campaign
  configuration are present.
