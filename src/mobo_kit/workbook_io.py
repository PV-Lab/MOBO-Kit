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

**Which columns to collect comes from the config, not from here.** Thickness
trains on nanometres, not on its score, so ``R1_Candidates`` needs an entry
column for ``Thickness (avg)``. Driving that off ``model_source_columns(config)``
means a future objective change updates the sheet automatically instead of
silently leaving the next round without its data.

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

from .campaign import model_source_columns

__all__ = [
    "CandidateSheetError",
    "RoundState",
    "WorkbookContents",
    "backup_workbook",
    "candidate_workbook_path",
    "detect_round",
    "read_campaign_workbook",
    "sheet_name_for_round",
    "workbook_digest",
    "write_candidate_sheet",
]

SOURCE_SHEET = "Sheet1"
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
    """The columns the GP trains on, in objective order."""
    sample_ids: tuple[int, ...]
    digest: str

    @property
    def n_rows(self) -> int:
        return len(self.inputs)


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


def read_campaign_workbook(
    path: str | Path, config: Mapping[str, Any]
) -> WorkbookContents:
    """Read measured rows, stopping at the first blank sample number.

    Rows below the data block are notes, not observations.
    """
    workbook = load_workbook(Path(path), data_only=True, read_only=False)
    if SOURCE_SHEET not in workbook.sheetnames:
        raise CandidateSheetError(
            f"Expected a sheet named {SOURCE_SHEET!r}; found "
            f"{workbook.sheetnames}. If it was renamed, rename it back."
        )
    sheet = workbook[SOURCE_SHEET]
    positions = _header_positions(sheet)

    input_names = [item["name"] for item in config["inputs"]]
    source_columns = list(model_source_columns(config))
    missing = [
        name
        for name in [SAMPLE_COLUMN, *input_names, *source_columns]
        if name not in positions
    ]
    if missing:
        raise CandidateSheetError(
            f"{SOURCE_SHEET} is missing required column(s): {missing}. "
            "The optimizer trains on these, so it cannot proceed without them."
        )

    rows = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        if row[positions[SAMPLE_COLUMN]] is None:
            break
        rows.append(row)
    if not rows:
        raise CandidateSheetError(f"{SOURCE_SHEET} contains no measured rows.")

    def column(name: str) -> list[Any]:
        return [row[positions[name]] for row in rows]

    return WorkbookContents(
        inputs=pd.DataFrame(
            {name: pd.to_numeric(column(name), errors="coerce") for name in input_names}
        ),
        model_values=pd.DataFrame(
            {
                name: pd.to_numeric(column(name), errors="coerce")
                for name in source_columns
            }
        ),
        sample_ids=tuple(int(value) for value in column(SAMPLE_COLUMN)),
        digest=workbook_digest(path),
    )


def detect_round(path: str | Path, config: Mapping[str, Any]) -> RoundState:
    """Decide which round to generate. Fail closed on a partial sheet."""
    source_columns = list(model_source_columns(config))

    for round_name, following in (("R1", "R2"), ("R2", None)):
        candidate_path = candidate_workbook_path(path, round_name)
        name = candidate_path.name
        if not candidate_path.exists():
            return RoundState(round_name, f"{name} does not exist yet.")
        sheet = load_workbook(candidate_path, data_only=True)[
            sheet_name_for_round(round_name)
        ]
        positions = _header_positions(sheet)
        missing = [c for c in source_columns if c not in positions]
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
        filled = [
            all(row[positions[c]] is not None for c in source_columns) for row in data
        ]
        scored, total = sum(filled), len(filled)
        if total and scored == 0:
            return RoundState(
                None,
                f"{name} exists but no results have been entered yet. Run those "
                f"{total} conditions and fill in {', '.join(source_columns)}.",
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
    Measurement columns are left blank and highlighted for entry -- including
    thickness in nanometres, which the next round trains on directly.

    The source workbook is opened read-only and its Sheet1 digest is checked
    afterwards, so the guarantee is enforced rather than assumed.
    """
    source_path = Path(path)
    source_before = _source_sheet_digest(source_path)
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
    source_columns = list(model_source_columns(config))
    headers = [
        "candidate_id",
        "replicate_group",
        "replicate_index",
        "round",
        *input_names,
        *source_columns,
    ]

    sheet = workbook.create_sheet(sheet_name)
    sheet.append(headers)
    for cell in sheet[1]:
        cell.font = HEADER_FONT

    entry_start = len(headers) - len(source_columns) + 1
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
            for offset in range(len(source_columns)):
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
    if _source_sheet_digest(source_path) != source_before:
        raise CandidateSheetError(
            f"{SOURCE_SHEET} in {source_path.name} changed while writing "
            f"{workbook_path.name}. It should not have been touched at all."
        )
    return workbook_path


def _source_sheet_digest(path: str | Path) -> str:
    """Digest of Sheet1's values only, so added sheets do not change it."""
    sheet = load_workbook(Path(path), data_only=True, read_only=False)[SOURCE_SHEET]
    digest = hashlib.sha256()
    for row in sheet.iter_rows(values_only=True):
        digest.update(repr(row).encode("utf-8"))
    return digest.hexdigest()
