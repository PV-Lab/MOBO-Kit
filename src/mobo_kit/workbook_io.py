"""Read the campaign workbook, write candidate sheets back.

The experimentalist's side of the loop: open ``Summary Table.xlsx``, fill in ten
inputs and the measurement columns, press one button, get a new sheet of
conditions to run.

Three rules this module keeps.

**The source workbook is never opened for writing.** Candidate sheets go to a
sibling file, ``<name>_R1_Candidates.xlsx``.

That is not the original plan, which was to add sheets to ``Summary Table.xlsx``
itself.  It changed because of a measured fact: **openpyxl discards cached
formula values on save.**  ``Uniformity score`` is a formula column
(``=L2*N2*O2``), so a single openpyxl round-trip turns it -- and every other
formula column -- into ``None`` for any reader that is not Excel, including this
one.  Verified directly: Z2:Z4 read ``[0.657, 0.587, 0.561]`` before a save that
only added an empty sheet, and ``[None, None, None]`` after.

Writing beside the workbook keeps the experimentalist's one-button flow (they
open the new file, fill it in, press the button again) and makes the read-only
invariant structural rather than merely asserted.

**Which columns to collect comes from the config, not from here.** Each
objective's ``measurement`` block names the raw columns its value is computed
from, so ``R1_Candidates`` asks for ``Coverage``, ``T1..T4`` and the rest rather
than for the three derived scores. Driving that off the config means a future
objective change updates the sheet automatically instead of silently leaving the
next round without its data.

**The derived scores are computed, not read.** Three of the workbook's score
cells are pasted literals that do not update when the measurements behind them
change, so :mod:`scores` recomputes all three and the stored cells become a
cross-check that warns. That is why ``model_values`` is keyed by objective name
and the stored cells appear separately as ``workbook_values``.

**Round detection is fail-closed.** A partially scored sheet is refused with a
plain sentence rather than being guessed at.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from .campaign import (
    measurement_entry_columns,
    measurement_specs,
    model_source_columns,
    objective_names,
    replicate_aggregates,
)
from .scores import (
    ScoreFinding,
    ScoreSeverity,
    compute_measurements,
    row_completeness,
)

__all__ = [
    "CandidateResults",
    "CandidateSheetError",
    "RoundState",
    "WorkbookContents",
    "backup_workbook",
    "candidate_workbook_path",
    "detect_round",
    "read_campaign_workbook",
    "read_candidate_results",
    "sheet_name_for_round",
    "source_sheet",
    "workbook_digest",
    "write_candidate_sheet",
]

#: Fallback for configs that predate `campaign.source_sheet`. The v4 workbook
#: names its sheets by round (`R0`, `R1`), so the sheet a campaign reads is now a
#: config key rather than a constant.
SOURCE_SHEET = "Sheet1"


def source_sheet(config: Mapping[str, Any]) -> str:
    """Which sheet holds the measured rows for this campaign."""
    return str((config.get("campaign") or {}).get("source_sheet", SOURCE_SHEET))


def formula_findings(
    path: str | Path, config: Mapping[str, Any]
) -> tuple[ScoreFinding, ...]:
    """Has a frozen score column's DEFINITION moved since it was recorded?

    A ``stored`` objective is read rather than recomputed, so nothing in Python
    knows what it means and no cross-check can catch a redefinition. This is the
    partial replacement: read the formula TEXT (never evaluate it) and compare it
    with the fingerprint in config.

    Requires a second read of the workbook with ``data_only=False``, because
    openpyxl gives either the formulas or their cached values and never both. That
    is why it is skipped entirely unless a fingerprint is declared.
    """
    from .campaign import measurement_specs, objective_names

    specs = list(measurement_specs(config))
    names = list(objective_names(config))
    wanted = [
        (name, spec)
        for name, spec in zip(names, specs)
        if spec is not None and spec.formula_fingerprint is not None
    ]
    if not wanted:
        return ()

    sheet_name = source_sheet(config)
    workbook = load_workbook(Path(path), data_only=False, read_only=False)
    if sheet_name not in workbook.sheetnames:
        return ()
    sheet = workbook[sheet_name]
    positions = _header_positions(sheet)

    findings: list[ScoreFinding] = []
    for name, spec in wanted:
        fingerprint = spec.formula_fingerprint
        column = fingerprint.column
        if column not in positions:
            findings.append(
                ScoreFinding(
                    severity=ScoreSeverity.WARNING,
                    code="fingerprint_column_absent",
                    objective=name,
                    row_position=-1,
                    sample_id=None,
                    message=(
                        f"{column!r} is not in {sheet_name}, so the frozen score's "
                        "definition cannot be checked at all."
                    ),
                    column=column,
                )
            )
            continue
        index = positions[column]
        seen: list[str] = []
        for row in sheet.iter_rows(min_row=2, values_only=True):
            if row[0] is None:
                break
            value = row[index]
            if isinstance(value, str) and value.startswith("="):
                seen.append(value)
        if not seen:
            findings.append(
                ScoreFinding(
                    severity=ScoreSeverity.WARNING,
                    code="fingerprint_no_formula",
                    objective=name,
                    row_position=-1,
                    sample_id=None,
                    message=(
                        f"{column!r} holds no formula on any row -- the values are "
                        "literals. A frozen score that is pasted rather than "
                        "computed cannot be checked against anything at all, which "
                        "is the one failure this contract cannot see."
                    ),
                    column=column,
                )
            )
            continue
        changed = [text for text in seen if not fingerprint.matches(text)]
        if changed:
            findings.append(
                ScoreFinding(
                    severity=ScoreSeverity.WARNING,
                    code="formula_fingerprint_changed",
                    objective=name,
                    row_position=-1,
                    sample_id=None,
                    message=(
                        f"{column!r} no longer matches the recorded definition. "
                        f"Recorded {fingerprint.formula!r}; found "
                        f"{changed[0]!r} (and {len(changed) - 1} other row(s) that "
                        "differ). This objective is READ, not recomputed, so the "
                        "change is not an error -- but every number computed under "
                        "the old definition is about a different quantity. Bump "
                        "objectives.contract_version and update the fingerprint."
                    ),
                    column=column,
                )
            )
        else:
            findings.append(
                ScoreFinding(
                    severity=ScoreSeverity.NOTE,
                    code="formula_fingerprint_unchanged",
                    objective=name,
                    row_position=-1,
                    sample_id=None,
                    message=(
                        f"{column!r} still computes {fingerprint.formula!r} on all "
                        f"{len(seen)} rows."
                    ),
                    column=column,
                )
            )
    return tuple(findings)
SAMPLE_COLUMN = "Sample number"
ENTRY_FILL = PatternFill("solid", fgColor="FFF2CC")
HEADER_FONT = Font(bold=True)


class CandidateSheetError(RuntimeError):
    """The workbook is not in a state this tool can act on."""


@dataclass(frozen=True)
class WorkbookContents:
    """Everything read out of the source sheet."""

    inputs: pd.DataFrame
    """Physical input values, columns in the config's declared order."""
    model_values: pd.DataFrame
    """What the GP trains on, one column per objective in objective order.

    Computed from the raw measurement columns for any objective that declares a
    ``measurement`` block; read from the declared column for one that does not.
    """
    workbook_values: pd.DataFrame
    """The stored derived cells, as the workbook holds them. Cross-check only."""
    inputs_used: pd.DataFrame
    """How many measured inputs each value came from -- 2 to 4 for thickness."""
    findings: tuple[ScoreFinding, ...]
    """Cross-check mismatches, excluded readings and disagreeing replicates."""
    sample_ids: tuple[int, ...]
    digest: str

    @property
    def n_rows(self) -> int:
        return len(self.inputs)

    @property
    def errors(self) -> tuple[ScoreFinding, ...]:
        from .scores import ScoreSeverity

        return tuple(f for f in self.findings if f.severity is ScoreSeverity.ERROR)

    @property
    def warnings(self) -> tuple[ScoreFinding, ...]:
        from .scores import ScoreSeverity

        return tuple(f for f in self.findings if f.severity is ScoreSeverity.WARNING)


@dataclass(frozen=True)
class RoundState:
    """Which round should be generated next, and why."""

    next_round: str | None
    reason: str
    scored_rows: int = 0
    total_rows: int = 0


def sheet_name_for_round(round_name: str) -> str:
    return f"{round_name.upper()}_Candidates"


def workbook_digest(path: str | Path) -> str:
    """SHA-256 of the whole file, for the unchanged-source proof."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backup_workbook(path: str | Path, *, timestamp: str | None = None) -> Path:
    """Copy the workbook next to itself before any write."""
    source = Path(path)
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = source.with_name(f"{source.stem}_backup_{stamp}{source.suffix}")
    shutil.copy2(source, destination)
    return destination


def _header_positions(sheet) -> dict[str, int]:
    header = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True))
    positions: dict[str, int] = {}
    for index, value in enumerate(header):
        if value is None:
            continue
        name = str(value).strip()
        # duplicate headers exist in this workbook; first occurrence wins and the
        # rest stay reachable by position
        positions.setdefault(name, index)
        # headers carry their unit inline ("precur_vol (uL)") while the config
        # declares name and unit separately; accept both spellings
        bare = name.split("(")[0].strip()
        if bare and bare != name:
            positions.setdefault(bare, index)
    return positions


def _near_misses(wanted: str, available: Sequence[str]) -> list[str]:
    """Headers that are plausibly the same column under a different name.

    Deliberately generous. The realistic cause of a missing column is not a typo
    in the sheet but a config describing a DIFFERENT campaign, where the same
    quantity was called something adjacent -- ``PL - Implied Voc (Max)`` against
    ``PL - Implied Voc (Max) Raw``. Prefix and containment catch that; edit
    distance would not, and would also match unrelated columns.
    """
    lowered = wanted.lower().strip()
    head = lowered.split("(")[0].strip()
    hits = [
        name
        for name in available
        if name.lower().strip() != lowered
        and (
            lowered in name.lower()
            or name.lower() in lowered
            or (len(head) > 3 and name.lower().startswith(head))
        )
    ]
    return hits[:4]


def _missing_columns_message(
    missing: Sequence[str],
    positions: Mapping[str, int],
    config: Mapping[str, Any],
    path: Path,
    sheet_name: str = SOURCE_SHEET,
) -> str:
    """Say which CONTRACT wanted the column, not just that it is absent.

    "Sheet1 is missing required column(s)" reads as a broken workbook, and the
    usual cause is the opposite: an intact workbook being read against another
    campaign's config. Naming the config and offering the near-miss headers turns
    a five-minute hunt into a glance.
    """
    campaign = config.get("campaign") or {}
    contract = (config.get("objectives") or {}).get("contract_version")
    lines = [
        f"{sheet_name} of {path.name} is missing column(s) that the campaign "
        f"configuration requires: {list(missing)}.",
        "",
        f"Configuration: {campaign.get('name')} "
        f"(status: {campaign.get('status')}, contract: {contract}).",
    ]
    if str(campaign.get("status")) == "archived":
        lines += [
            "",
            "THAT CONFIGURATION IS ARCHIVED. It describes a previous campaign, "
            "whose workbook had different columns, so this is almost certainly a "
            "config/workbook mismatch rather than a problem with the workbook. "
            "Point the launcher at the active campaign configuration instead.",
        ]
    suggestions = {
        name: _near_misses(name, list(positions)) for name in missing
    }
    named = {name: hits for name, hits in suggestions.items() if hits}
    if named:
        lines += ["", "The sheet does have these, which look related:"]
        for name, hits in named.items():
            lines.append(f"  wanted {name!r} -> found {hits}")
        lines += [
            "",
            "If one of those is the same measurement under a new name, the fix is "
            "a `measurement` column in the config, not an edit to the workbook.",
        ]
    return "\n".join(lines)


def read_campaign_workbook(
    path: str | Path, config: Mapping[str, Any]
) -> WorkbookContents:
    """Read measured rows, stopping at the first blank sample number.

    Rows below the data block are notes, not observations.
    """
    sheet_name = source_sheet(config)
    workbook = load_workbook(Path(path), data_only=True, read_only=False)
    if sheet_name not in workbook.sheetnames:
        raise CandidateSheetError(
            f"This campaign reads its measured rows from a sheet named "
            f"{sheet_name!r} (campaign.source_sheet); {Path(path).name} has "
            f"{workbook.sheetnames}. Either the workbook is for a different "
            "campaign, or the sheet was renamed."
        )
    sheet = workbook[sheet_name]
    positions = _header_positions(sheet)

    input_names = [item["name"] for item in config["inputs"]]
    specs = measurement_specs(config)
    computed = [spec for spec in specs if spec is not None]
    declared = list(model_source_columns(config))
    names = list(objective_names(config))
    required_entry, optional_entry = measurement_entry_columns(config)

    missing = [
        name
        for name in [SAMPLE_COLUMN, *input_names, *required_entry]
        if name not in positions
    ]
    if missing:
        raise CandidateSheetError(
            _missing_columns_message(
                missing, positions, config, Path(path), sheet_name
            )
        )

    rows = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        if row[positions[SAMPLE_COLUMN]] is None:
            break
        rows.append(row)
    if not rows:
        raise CandidateSheetError(f"{sheet_name} contains no measured rows.")

    def column(name: str) -> list[Any]:
        return [row[positions[name]] for row in rows]

    # every column any recipe or cross-check may look at, kept as raw cells:
    # `scores` is the one place that knows how this workbook spells "not measured"
    wanted: list[str] = [*required_entry, *optional_entry, *declared]
    for spec in computed:
        wanted.extend(check.column for check in spec.cross_checks)
    raw = pd.DataFrame(
        {
            name: column(name)
            for name in dict.fromkeys(wanted)
            if name in positions
        },
        dtype=object,
    )
    sample_ids = tuple(int(value) for value in column(SAMPLE_COLUMN))

    model_frame: dict[str, Any] = {}
    findings: tuple[ScoreFinding, ...] = ()
    inputs_used = pd.DataFrame(index=range(len(rows)))
    if computed:
        result = compute_measurements(raw, computed, sample_ids=sample_ids)
        findings = result.findings
        inputs_used = result.inputs_used
        for name in result.values.columns:
            model_frame[name] = result.values[name]
    for name, spec, declared_column in zip(names, specs, declared):
        if spec is None:
            model_frame[name] = pd.to_numeric(
                column(declared_column), errors="coerce"
            )

    findings = tuple(findings) + formula_findings(path, config)

    return WorkbookContents(
        inputs=pd.DataFrame(
            {name: pd.to_numeric(column(name), errors="coerce") for name in input_names}
        ),
        model_values=pd.DataFrame({name: model_frame[name] for name in names}),
        workbook_values=pd.DataFrame(
            {
                name: pd.to_numeric(column(name), errors="coerce")
                for name in dict.fromkeys(declared)
                if name in positions
            }
        ),
        inputs_used=inputs_used,
        findings=findings,
        sample_ids=sample_ids,
        digest=workbook_digest(path),
    )


@dataclass(frozen=True)
class CandidateResults:
    """Measurements read back out of one round's candidate sheet.

    The films of one condition are separate experimental rows but one design
    point, so they are aggregated to a single observation before the next round
    trains on them.  ``replicate_spread`` keeps the within-condition scatter that
    aggregation discards -- that is the raw material for ``train_Yvar``.
    """

    round_name: str
    conditions: pd.DataFrame
    """One row per condition, input columns in the config's declared order."""
    model_values: pd.DataFrame
    """One row per condition, one column per objective. Aggregated."""
    replicates: pd.DataFrame
    """One row per film: candidate_id, replicate_index, then objective values."""
    replicate_spread: pd.DataFrame
    """Per-condition sd in each objective's aggregation space. NaN below 2 films."""
    films_used: pd.DataFrame
    """How many films each condition's value was aggregated from."""
    findings: tuple[ScoreFinding, ...]
    candidate_ids: tuple[str, ...]

    @property
    def n_conditions(self) -> int:
        return len(self.conditions)

    @property
    def errors(self) -> tuple[ScoreFinding, ...]:
        return tuple(f for f in self.findings if f.severity is ScoreSeverity.ERROR)


def _aggregate(values: np.ndarray, rule: str) -> tuple[float, float]:
    """Collapse one condition's film values to (observation, spread).

    Spread is the sample sd in the aggregation space: for ``mean`` a sd in the
    objective's own units, for ``mean_of_log`` a sd of ``log`` values.  Either is
    what ``train_Yvar`` wants only when it matches the space the GP trains in,
    which :func:`campaign.replicate_aggregates` enforces.  It is NaN for a single
    film, which is honest: one film measures no reproducibility at all.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    if rule == "mean_of_log":
        if np.any(finite <= 0):
            raise CandidateSheetError(
                "mean_of_log aggregation needs strictly positive values; got "
                f"{finite.tolist()}."
            )
        logs = np.log(finite)
        spread = float(np.std(logs, ddof=1)) if finite.size > 1 else float("nan")
        return float(np.exp(logs.mean())), spread
    spread = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
    return float(finite.mean()), spread


def read_candidate_results(
    path: str | Path, config: Mapping[str, Any], round_name: str
) -> CandidateResults:
    """Read a filled-in candidate sheet and aggregate it to design points.

    ``path`` is the SOURCE workbook; the candidate sheet is found beside it, the
    same way :func:`write_candidate_sheet` put it there.  Objective values are
    computed per film by :mod:`scores` -- the same recipes the source sheet uses,
    so R0 and R1 observations are commensurable -- and then aggregated per
    ``replicate_group``.
    """
    candidate_path = candidate_workbook_path(path, round_name)
    if not candidate_path.exists():
        raise CandidateSheetError(
            f"{candidate_path.name} does not exist, so there are no {round_name} "
            "measurements to read."
        )
    sheet_name = sheet_name_for_round(round_name)
    workbook = load_workbook(candidate_path, data_only=True)
    if sheet_name not in workbook.sheetnames:
        raise CandidateSheetError(
            f"{candidate_path.name} has no {sheet_name!r} sheet; found "
            f"{workbook.sheetnames}."
        )
    sheet = workbook[sheet_name]
    positions = _header_positions(sheet)

    input_names = [item["name"] for item in config["inputs"]]
    names = list(objective_names(config))
    specs = [spec for spec in measurement_specs(config) if spec is not None]
    rules = list(replicate_aggregates(config))
    required_entry, optional_entry = measurement_entry_columns(config)

    missing = [
        column
        for column in ["candidate_id", *input_names, *required_entry]
        if column not in positions
    ]
    if missing:
        raise CandidateSheetError(
            f"{candidate_path.name} is missing column(s) {missing}. It was "
            "probably created by an older version; regenerate it."
        )

    rows = [
        row
        for row in sheet.iter_rows(min_row=2, values_only=True)
        if row[positions["candidate_id"]] is not None
    ]
    if not rows:
        raise CandidateSheetError(f"{sheet_name} contains no candidate rows.")

    def column(name: str) -> list[Any]:
        return [row[positions[name]] for row in rows]

    group_column = "replicate_group" if "replicate_group" in positions else "candidate_id"
    groups = [str(value) for value in column(group_column)]
    film_labels = [str(value) for value in column("candidate_id")]

    wanted = [*required_entry, *optional_entry]
    for spec in specs:
        wanted.extend(check.column for check in spec.cross_checks)
    raw = pd.DataFrame(
        {name: column(name) for name in dict.fromkeys(wanted) if name in positions},
        dtype=object,
    )
    per_film = compute_measurements(raw, specs, sample_ids=film_labels)
    findings = list(per_film.findings)

    inputs = pd.DataFrame(
        {name: pd.to_numeric(column(name), errors="coerce") for name in input_names}
    )

    ordered_groups = list(dict.fromkeys(groups))
    group_index = pd.Series(groups)

    condition_rows: list[dict[str, float]] = []
    value_rows: list[dict[str, float]] = []
    spread_rows: list[dict[str, float]] = []
    count_rows: list[dict[str, int]] = []
    for group in ordered_groups:
        mask = (group_index == group).to_numpy()
        block = inputs.loc[mask]
        first = block.iloc[0]
        for name in input_names:
            if not np.allclose(
                block[name].to_numpy(dtype=float), float(first[name]), equal_nan=True
            ):
                raise CandidateSheetError(
                    f"The films of {group} do not share the same {name}. Replicates "
                    "must be the same recipe; edit the sheet or regenerate it."
                )
        condition_rows.append({name: float(first[name]) for name in input_names})

        values: dict[str, float] = {}
        spreads: dict[str, float] = {}
        counts: dict[str, int] = {}
        for name, rule in zip(names, rules):
            film_values = per_film.values.loc[mask, name].to_numpy(dtype=float)
            observation, spread = _aggregate(film_values, rule)
            values[name] = observation
            spreads[name] = spread
            counts[name] = int(np.isfinite(film_values).sum())
            if counts[name] == 0:
                findings.append(
                    ScoreFinding(
                        severity=ScoreSeverity.ERROR,
                        code="condition_has_no_usable_film",
                        objective=name,
                        row_position=ordered_groups.index(group),
                        sample_id=group,
                        message=(
                            f"none of the {int(mask.sum())} films of {group} produced "
                            f"a usable {name} value."
                        ),
                    )
                )
        value_rows.append(values)
        spread_rows.append(spreads)
        count_rows.append(counts)

    replicates = pd.DataFrame(
        {
            "candidate_id": film_labels,
            "replicate_group": groups,
            **(
                {"replicate_index": pd.to_numeric(column("replicate_index"))}
                if "replicate_index" in positions
                else {}
            ),
            **{name: per_film.values[name] for name in names},
        }
    )

    return CandidateResults(
        round_name=round_name.upper(),
        conditions=pd.DataFrame(condition_rows, columns=input_names),
        model_values=pd.DataFrame(value_rows, columns=names),
        replicates=replicates,
        replicate_spread=pd.DataFrame(spread_rows, columns=names),
        films_used=pd.DataFrame(count_rows, columns=names),
        findings=tuple(findings),
        candidate_ids=tuple(ordered_groups),
    )


def detect_round(path: str | Path, config: Mapping[str, Any]) -> RoundState:
    """Decide which round to generate. Fail closed on a partial sheet.

    "Measured" is a per-objective question once objectives are computed rather
    than read: ``product`` and ``log10_product`` need every input, while
    thickness needs only one of ``T1..T4``. Requiring all four would report a
    finished sheet as partial -- nine of the fifteen R0 rows have two readings.
    """
    specs = [spec for spec in measurement_specs(config) if spec is not None]
    required_entry, optional_entry = measurement_entry_columns(config)

    for round_name, following in (("R1", "R2"), ("R2", None)):
        candidate_path = candidate_workbook_path(path, round_name)
        name = candidate_path.name
        if not candidate_path.exists():
            return RoundState(round_name, f"{name} does not exist yet.")
        sheet = load_workbook(candidate_path, data_only=True)[
            sheet_name_for_round(round_name)
        ]
        positions = _header_positions(sheet)
        missing = [c for c in required_entry if c not in positions]
        if missing:
            raise CandidateSheetError(
                f"{name} is missing entry column(s) {missing}. It was probably "
                "created by an older version; delete the sheet and regenerate it."
            )
        data = [
            row
            for row in sheet.iter_rows(min_row=2, values_only=True)
            if any(value is not None for value in row)
        ]
        if specs:
            frame = pd.DataFrame(
                {
                    entry: [row[positions[entry]] for row in data]
                    for entry in dict.fromkeys((*required_entry, *optional_entry))
                    if entry in positions
                },
                dtype=object,
                index=range(len(data)),
            )
            filled = list(row_completeness(frame, specs))
        else:
            filled = [
                all(row[positions[c]] is not None for c in required_entry)
                for row in data
            ]
        scored, total = sum(filled), len(filled)
        if total and scored == 0:
            return RoundState(
                None,
                f"{name} exists but no results have been entered yet. Run those "
                f"{total} conditions and fill in {', '.join(required_entry)}.",
                scored,
                total,
            )
        if scored < total:
            return RoundState(
                None,
                f"{name} is partly filled in: {scored} of {total} rows have all "
                "measurements. Complete the remaining rows, or clear them, then "
                "try again.",
                scored,
                total,
            )
        if following is None:
            return RoundState(
                None,
                "R1 and R2 are both complete. The campaign is finished.",
                scored,
                total,
            )
    return RoundState(None, "Nothing to do.")


def candidate_workbook_path(path: str | Path, round_name: str) -> Path:
    """Where this round's candidates are written, beside the source workbook."""
    source = Path(path)
    return source.with_name(f"{source.stem}_{sheet_name_for_round(round_name)}.xlsx")


def write_candidate_sheet(
    path: str | Path,
    config: Mapping[str, Any],
    conditions: pd.DataFrame,
    *,
    round_name: str,
    replicates: int = 3,
    make_backup: bool = True,
) -> Path:
    """Write this round's worklist to a sibling workbook.

    One row per physical film, three per condition sharing a ``replicate_group``.
    Measurement columns are left blank and highlighted for entry -- the raw
    columns each objective is computed from, not the derived scores, because the
    scores are now computed in Python.

    Optional entry columns (``T3``, ``T4``, ``T anom``) are offered but not
    demanded: a film with two thickness readings is complete.

    The source workbook is opened read-only and its Sheet1 digest is checked
    afterwards, so the guarantee is enforced rather than assumed.
    """
    source_path = Path(path)
    source_before = _source_sheet_digest(source_path, source_sheet(config))
    workbook_path = candidate_workbook_path(source_path, round_name)
    sheet_name = sheet_name_for_round(round_name)
    if workbook_path.exists():
        if make_backup:
            backup_workbook(workbook_path)
        raise CandidateSheetError(
            f"{workbook_path.name} already exists. Rename or delete it first; "
            "this tool does not overwrite a file that may hold measurements."
        )
    from openpyxl import Workbook

    workbook = Workbook()
    workbook.remove(workbook.active)

    input_names = [item["name"] for item in config["inputs"]]
    required_entry, optional_entry = measurement_entry_columns(config)
    entry_names = [*required_entry, *optional_entry]
    headers = [
        "candidate_id",
        "replicate_group",
        "replicate_index",
        "round",
        *input_names,
        *entry_names,
    ]

    sheet = workbook.create_sheet(sheet_name)
    sheet.append(headers)
    for cell in sheet[1]:
        cell.font = HEADER_FONT

    entry_start = len(headers) - len(entry_names) + 1
    for index, (_, condition) in enumerate(conditions.iterrows(), start=1):
        candidate_id = f"{round_name.upper()}_C{index:02d}"
        for replicate in range(1, replicates + 1):
            sheet.append(
                [
                    candidate_id,
                    candidate_id,
                    replicate,
                    round_name.upper(),
                    *[float(condition[name]) for name in input_names],
                ]
            )
            for offset in range(len(entry_names)):
                sheet.cell(row=sheet.max_row, column=entry_start + offset).fill = (
                    ENTRY_FILL
                )

    for index, header in enumerate(headers, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = max(
            12, min(24, len(header) + 3)
        )
    sheet.freeze_panes = "A2"

    workbook.save(workbook_path)

    # the invariant, actually enforced rather than asserted
    if _source_sheet_digest(source_path, source_sheet(config)) != source_before:
        raise CandidateSheetError(
            f"{source_sheet(config)} in {source_path.name} changed while writing "
            f"{workbook_path.name}. It should not have been touched at all."
        )
    return workbook_path


def _source_sheet_digest(path: str | Path, sheet_name: str = SOURCE_SHEET) -> str:
    """Digest of the source sheet's values only, so added sheets do not change it."""
    sheet = load_workbook(Path(path), data_only=True, read_only=False)[sheet_name]
    digest = hashlib.sha256()
    for row in sheet.iter_rows(values_only=True):
        digest.update(repr(row).encode("utf-8"))
    return digest.hexdigest()
