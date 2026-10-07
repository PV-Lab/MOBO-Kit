# src/utils.py
from __future__ import annotations
from dataclasses import dataclass
import csv
import io
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import pandas as pd
import numpy as np
import torch
import yaml

from .design import Design


_METADATA_LABELS = ("units", "start", "stop", "step")
_CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


@dataclass
class ParsedCampaignCSV:
    """Parsed metadata-style campaign CSV and its inferred data contract.

    ``metadata_row_count`` counts rows between the header and the first
    experimental row, including an optional blank separator. Header positions
    in ``duplicate_headers`` are one-based CSV/Excel column positions.
    """

    config: dict[str, Any]
    data: pd.DataFrame
    input_columns: list[str]
    objective_columns: list[str]
    metadata_row_count: int
    raw_headers: list[str]
    duplicate_headers: dict[str, list[int]]
    encoding: str


def _read_raw_csv(path: str | Path) -> tuple[Path, list[list[str]], str]:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Campaign CSV not found: {csv_path}")

    payload = csv_path.read_bytes()
    if not payload:
        raise ValueError(f"Campaign CSV is empty: {csv_path}")

    decode_errors: list[str] = []
    for encoding in _CSV_ENCODINGS:
        try:
            text = payload.decode(encoding)
        except UnicodeDecodeError as exc:
            decode_errors.append(f"{encoding}: {exc}")
            continue

        try:
            rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
        except csv.Error as exc:
            raise ValueError(f"Malformed CSV syntax in {csv_path}: {exc}") from exc

        if not rows or not any(cell.strip() for cell in rows[0]):
            raise ValueError(f"Campaign CSV has no usable header row: {csv_path}")
        return csv_path, rows, encoding

    detail = "; ".join(decode_errors)
    raise UnicodeError(
        f"Could not decode campaign CSV {csv_path} using "
        f"{', '.join(_CSV_ENCODINGS)}. {detail}"
    )


def _normalize_row_widths(
    rows: list[list[str]], width: int, csv_path: Path
) -> list[list[str]]:
    normalized: list[list[str]] = []
    for line_number, row in enumerate(rows, start=1):
        if len(row) > width and any(cell.strip() for cell in row[width:]):
            raise ValueError(
                f"CSV row {line_number} in {csv_path} has {len(row)} fields but "
                f"the header has {width}; extra nonblank fields are not allowed."
            )
        normalized.append((row[:width] + [""] * width)[:width])
    return normalized


def _duplicate_header_positions(headers: Sequence[str]) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {}
    for position, raw_header in enumerate(headers, start=1):
        header = raw_header.strip()
        if header:
            positions.setdefault(header, []).append(position)
    return {name: found for name, found in positions.items() if len(found) > 1}


def _find_metadata_rows(
    rows: Sequence[Sequence[str]], csv_path: Path
) -> tuple[dict[str, int], int]:
    found: dict[str, int] = {}
    label_columns: set[int] = set()

    for row_index, row in enumerate(rows[1:], start=1):
        matches = [
            (column_index, cell.strip().lower())
            for column_index, cell in enumerate(row)
            if cell.strip().lower() in _METADATA_LABELS
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Metadata row {row_index + 1} in {csv_path} contains more than "
                f"one metadata label: {[label for _, label in matches]}"
            )
        if not matches:
            continue

        column_index, label = matches[0]
        if label in found:
            raise ValueError(
                f"Duplicate '{label}' metadata row in {csv_path} "
                f"(rows {found[label] + 1} and {row_index + 1})."
            )
        found[label] = row_index
        label_columns.add(column_index)
        if len(found) == len(_METADATA_LABELS):
            break

    missing = [label for label in _METADATA_LABELS if label not in found]
    if missing:
        raise ValueError(
            "Plain data CSVs are not supported by parse_campaign_csv; expected "
            "labeled units/start/stop/step metadata rows. "
            f"Missing metadata rows: {missing}."
        )
    if len(label_columns) != 1:
        raise ValueError(
            f"Metadata labels in {csv_path} must use one consistent label column; "
            f"found columns {[index + 1 for index in sorted(label_columns)]}."
        )

    return found, next(iter(label_columns))


def _parse_metadata_number(
    value: str, *, label: str, header: str, position: int
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Malformed {label} metadata for input '{header}' at column "
            f"{position}: {value!r} is not numeric."
        ) from exc
    if not np.isfinite(number):
        raise ValueError(
            f"Malformed {label} metadata for input '{header}' at column "
            f"{position}: values must be finite."
        )
    return number


def parse_campaign_csv(
    path: str | Path,
    expected_objectives: Sequence[str] | None = None,
) -> ParsedCampaignCSV:
    """Parse a metadata-style campaign CSV without fixed row offsets.

    Inputs are named columns with complete numeric ``start``, ``stop``, and
    ``step`` metadata. If ``expected_objectives`` is omitted, objectives are
    inferred from named non-input columns after an optional blank separator.
    Plain data-only CSVs and duplicate named headers are rejected explicitly.
    Experimental values are preserved here and validated numerically by
    :func:`split_XY` at the model boundary.
    """

    csv_path, raw_rows, encoding = _read_raw_csv(path)
    raw_headers = list(raw_rows[0])
    rows = _normalize_row_widths(raw_rows, len(raw_headers), csv_path)
    headers = [header.strip() for header in raw_headers]

    duplicate_headers = _duplicate_header_positions(raw_headers)
    if duplicate_headers:
        rendered = ", ".join(
            f"{name!r} at columns {positions}"
            for name, positions in duplicate_headers.items()
        )
        raise ValueError(
            "Duplicate CSV headers detected before pandas column renaming: "
            f"{rendered}."
        )

    metadata_rows, metadata_label_column = _find_metadata_rows(rows, csv_path)
    last_metadata_row = max(metadata_rows.values())
    data_start = last_metadata_row + 1
    while data_start < len(rows) and not any(cell.strip() for cell in rows[data_start]):
        data_start += 1
    if data_start >= len(rows):
        raise ValueError(f"Campaign CSV has an empty experimental section: {csv_path}")

    input_columns: list[str] = []
    input_positions: list[int] = []
    input_specs: list[dict[str, Any]] = []

    for column_index, header in enumerate(headers):
        if not header or column_index == metadata_label_column:
            continue

        numeric_values = {
            label: rows[row_index][column_index].strip()
            for label, row_index in metadata_rows.items()
            if label in {"start", "stop", "step"}
        }
        present = {label: bool(value) for label, value in numeric_values.items()}
        if not any(present.values()):
            continue
        if not all(present.values()):
            missing = [label for label, is_present in present.items() if not is_present]
            raise ValueError(
                f"Incomplete numeric metadata for column '{header}': missing "
                f"{missing}. Input columns require start, stop, and step."
            )

        start = _parse_metadata_number(
            numeric_values["start"],
            label="start",
            header=header,
            position=column_index + 1,
        )
        stop = _parse_metadata_number(
            numeric_values["stop"],
            label="stop",
            header=header,
            position=column_index + 1,
        )
        step = _parse_metadata_number(
            numeric_values["step"],
            label="step",
            header=header,
            position=column_index + 1,
        )
        if stop < start:
            raise ValueError(
                f"Invalid metadata for input '{header}': stop ({stop}) is below "
                f"start ({start})."
            )
        if step <= 0:
            raise ValueError(
                f"Invalid metadata for input '{header}': step must be positive."
            )

        unit = rows[metadata_rows["units"]][column_index].strip() or None
        input_columns.append(header)
        input_positions.append(column_index)
        input_specs.append(
            {
                "name": header,
                "unit": unit,
                "start": start,
                "stop": stop,
                "step": step,
            }
        )

    if not input_columns:
        raise ValueError(
            "Campaign CSV contains no input columns with complete numeric "
            f"metadata: {csv_path}"
        )

    named_non_input_positions = [
        column_index
        for column_index, header in enumerate(headers)
        if header
        and column_index != metadata_label_column
        and column_index not in input_positions
    ]

    if expected_objectives is not None:
        if isinstance(expected_objectives, str):
            objective_columns = [expected_objectives.strip()]
        else:
            objective_columns = [str(name).strip() for name in expected_objectives]
        if not objective_columns or any(not name for name in objective_columns):
            raise ValueError(
                "expected_objectives must contain at least one nonblank name."
            )
        if len(set(objective_columns)) != len(objective_columns):
            raise ValueError("expected_objectives contains duplicate names.")
        missing_objectives = [name for name in objective_columns if name not in headers]
        if missing_objectives:
            raise ValueError(
                f"Missing objective columns in campaign CSV: {missing_objectives}."
            )
        input_objectives = [name for name in objective_columns if name in input_columns]
        if input_objectives:
            raise ValueError(
                f"Columns cannot be both inputs and objectives: {input_objectives}."
            )
    else:
        objective_positions = [
            position
            for position in named_non_input_positions
            if position > max(input_positions)
        ]
        separator_positions = [
            column_index
            for column_index, header in enumerate(headers)
            if not header and column_index > max(input_positions)
        ]
        if separator_positions:
            objective_positions = [
                position
                for position in objective_positions
                if position > separator_positions[0]
            ]
        objective_columns = [headers[position] for position in objective_positions]

    if not objective_columns:
        raise ValueError(
            "Campaign CSV has no named objective columns. Provide objective headers "
            "or pass expected_objectives."
        )

    unnamed_positions = [
        column_index for column_index, header in enumerate(headers) if not header
    ]
    experimental_rows: list[list[str]] = []
    for source_row, row in enumerate(rows[data_start:], start=data_start + 1):
        if not any(cell.strip() for cell in row):
            continue
        populated_unnamed = [
            column_index + 1
            for column_index in unnamed_positions
            if row[column_index].strip()
        ]
        if populated_unnamed:
            raise ValueError(
                f"Experimental row {source_row} contains values in unnamed columns "
                f"{populated_unnamed}; add explicit headers before parsing."
            )
        experimental_rows.append(row)

    if not experimental_rows:
        raise ValueError(f"Campaign CSV has an empty experimental section: {csv_path}")

    data_positions = [
        column_index
        for column_index, header in enumerate(headers)
        if header and column_index != metadata_label_column
    ]
    data = pd.DataFrame(
        [
            [
                row[column_index] if row[column_index].strip() else pd.NA
                for column_index in data_positions
            ]
            for row in experimental_rows
        ],
        columns=[headers[column_index] for column_index in data_positions],
    )

    config = {
        "inputs": input_specs,
        "objectives": {"names": objective_columns},
        "constraints": [],
    }
    return ParsedCampaignCSV(
        config=config,
        data=data,
        input_columns=input_columns,
        objective_columns=objective_columns,
        metadata_row_count=data_start - 1,
        raw_headers=raw_headers,
        duplicate_headers=duplicate_headers,
        encoding=encoding,
    )


def get_objective_names(cfg: dict) -> List[str]:
    names = cfg.get("objectives", {}).get("names", [])
    if not names:
        raise ValueError("Config must have objectives.names as a non-empty list.")
    return list(names)


def load_csv(
    path: str | Path,
    expected_objectives: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load only experimental rows from a metadata-style campaign CSV."""

    return parse_campaign_csv(path, expected_objectives=expected_objectives).data


def _blank_mask(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.isna() | frame.apply(
        lambda column: column.map(
            lambda value: isinstance(value, str) and not value.strip()
        )
    )


def _numeric_model_frame(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    blank = _blank_mask(frame)
    converted = frame.apply(pd.to_numeric, errors="coerce")
    invalid = converted.isna() | ~np.isfinite(converted.astype(float))
    if invalid.any().any():
        locations = [
            f"row {frame.index[row]!r}, column {frame.columns[column]!r}"
            for row, column in zip(*np.where(invalid.to_numpy()))
        ]
        detail = ", ".join(locations[:8])
        if len(locations) > 8:
            detail += f", and {len(locations) - 8} more"
        kind = "blank" if (invalid & blank).any().any() else "nonnumeric"
        raise ValueError(
            f"{role} model data contains {kind} or non-finite values at {detail}."
        )
    return converted.astype(float)


def split_XY(
    df: pd.DataFrame, design: Design, config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select and validate named numeric model inputs and objectives.

    Rows are never filled with zero or silently discarded. Any incomplete row
    must be completed or removed explicitly by a campaign/QC policy before this
    model-boundary function is called.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("split_XY expects a pandas.DataFrame.")
    if df.empty:
        raise ValueError("Experimental data is empty; no model rows are available.")
    if df.columns.duplicated().any():
        duplicates = list(dict.fromkeys(df.columns[df.columns.duplicated()].tolist()))
        raise ValueError(f"Experimental DataFrame has duplicate columns: {duplicates}.")

    x_cols = list(design.names)
    y_cols = get_objective_names(config)
    miss_x = [column for column in x_cols if column not in df.columns]
    miss_y = [column for column in y_cols if column not in df.columns]
    if miss_x or miss_y:
        parts = []
        if miss_x:
            parts.append(f"missing inputs: {miss_x}")
        if miss_y:
            parts.append(f"missing objectives: {miss_y}")
        raise KeyError("CSV column check failed: " + "; ".join(parts))

    X_raw = df.loc[:, x_cols].copy()
    Y_raw = df.loc[:, y_cols].copy()
    objective_blanks = _blank_mask(Y_raw)
    if objective_blanks.all().all():
        raise ValueError(
            "All objective values are blank; no completed model rows exist."
        )

    all_blank_rows = objective_blanks.all(axis=1)
    if all_blank_rows.any():
        raise ValueError(
            "Objective values are blank for rows "
            f"{Y_raw.index[all_blank_rows].tolist()}; rows are not dropped silently."
        )
    partial_rows = objective_blanks.any(axis=1)
    if partial_rows.any():
        raise ValueError(
            "Partially completed objective rows found at indices "
            f"{Y_raw.index[partial_rows].tolist()}; complete every objective "
            "before modeling."
        )

    X = _numeric_model_frame(X_raw, "Input")
    Y = _numeric_model_frame(Y_raw, "Objective")
    return X, Y


def select_device(prefer: str = "cuda") -> torch.device:
    return torch.device(
        "cuda" if prefer == "cuda" and torch.cuda.is_available() else "cpu"
    )


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def np_to_torch(
    *arrays: np.ndarray | pd.DataFrame,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    return_device: bool = False,
):
    """
    Convert one or more NumPy arrays to torch tensors on the chosen device.

    Usage:
        X_t = np_to_torch(X)                                       # one array -> one tensor
        X_t, Y_t = np_to_torch(X, Y)                               # many arrays -> many tensors
        (X_t, Y_t), dev = np_to_torch(X, Y, return_device=True)    # also get the device used
    """
    if device is None:
        device = select_device("cuda")  # uses your existing helper
    tensors = tuple(
        torch.as_tensor(np.asarray(array), dtype=dtype, device=device)
        for array in arrays
    )
    out = tensors[0] if len(tensors) == 1 else tensors
    return (out, device) if return_device else out


def torch_to_np(*tensors: torch.Tensor):
    """
    Convert one or more torch tensors to NumPy arrays (detached, moved to CPU).

    Usage:
        X_np = torch_to_np(X_t)
        X_np, Y_np = torch_to_np(X_t, Y_t)
    """
    arrays = tuple(t.detach().cpu().numpy() for t in tensors)
    return arrays[0] if len(arrays) == 1 else arrays


def csv_to_config(
    csv_path: str | Path,
    output_path: str | Path | None = None,
    expected_objectives: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a configuration from a metadata-style campaign CSV.

    This wrapper has no filesystem side effect unless ``output_path`` is
    supplied explicitly. Generic campaign conversion always defaults to no
    constraints.
    """

    config = parse_campaign_csv(
        csv_path, expected_objectives=expected_objectives
    ).config
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(
                config,
                stream,
                default_flow_style=False,
                sort_keys=False,
                indent=2,
            )
    return config
