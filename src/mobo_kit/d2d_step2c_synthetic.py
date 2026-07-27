"""Sanitized, test-only end-to-end fixture for the Step 2C fast pipeline."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from openpyxl import Workbook
from openpyxl.utils.cell import range_boundaries
import yaml

from .d2d_campaign import D2D_WORKBOOK_INPUT_COLUMNS, sha256_file
from .d2d_step2c_config import (
    ExecutionModeSettings,
    ResolvedStep2CConfig,
    load_step2c_config,
)
from .d2d_step2c_robustness import (
    Step2CRobustnessResult,
    _run_d2d_step2c_robustness_resolved,
)


SYNTHETIC_STEP2C_HEADERS = (
    "Sample number",
    *D2D_WORKBOOK_INPUT_COLUMNS,
    "Coverage",
    "Uniformity",
    "1 - Uniformity",
    "Phase purity",
    "PL - Implied Voc (Max)",
    "Photoconductance (Max)",
    "Log10 (Photoconductance (Max) x PL - Implied Voc (Max))",
    "T1",
    "T2",
    "T3",
    "T4",
    "T anom",
    "Thickness (avg)",
    "Normalized thickness (sigma = 250)",
    "Uniformity score",
    "Optoelectronic score",
    "Thickness score",
    "Stability score?",
    "Total combination - addition",
    "Total combination - multiplied",
    "Uniformity score absolute difference",
    "Optoelectronic score absolute difference",
    "Thickness absolute difference",
    "Total score absolute difference",
)


def _canonical_hash(value: dict) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _synthetic_off_grid_exception(
    config: ResolvedStep2CConfig,
):
    """Return a deterministic, non-campaign off-grid fixture exception."""

    if len(config.off_grid_exceptions) != 1:
        raise ValueError("Synthetic Step 2C requires one inherited off-grid exception.")
    inherited = config.off_grid_exceptions[0]
    dimension = tuple(config.design.names).index(inherited.input_name)
    grid = np.asarray(config.design.var_array[dimension], dtype=float)
    observed = float(inherited.observed_value)
    if np.any(np.isclose(observed, grid, rtol=0.0, atol=1.0e-12)):
        raise RuntimeError("Synthetic sentinel unexpectedly lies on the input grid.")
    return replace(
        inherited,
        observed_value=observed,
        reason="sanitized synthetic off-grid fixture",
    )


def _synthetic_row(config: ResolvedStep2CConfig, sample_index: int) -> list[object]:
    sample_id = config.expected_sample_ids[sample_index]
    physical: list[float] = []
    for dimension, grid in enumerate(config.design.var_array):
        grid_rng = np.random.default_rng(20_260_300 + dimension)
        grid_index = int(grid_rng.integers(0, len(grid), size=15)[sample_index])
        physical.append(float(grid[grid_index]))
    exception = _synthetic_off_grid_exception(config)
    if sample_id == exception.sample_id:
        dimension = tuple(config.design.names).index(exception.input_name)
        physical[dimension] = exception.observed_value

    normalized_physical = np.asarray(
        [
            (value - float(np.min(grid))) / (float(np.max(grid)) - float(np.min(grid)))
            for value, grid in zip(physical, config.design.var_array, strict=True)
        ],
        dtype=float,
    )
    # Smooth public-CI responses keep every strict leave-one-out GP fold
    # deterministic across supported platforms.
    smooth_response_weights = np.asarray(
        [0.31, 0.23, 0.17, 0.11, 0.07, 0.05, 0.03, 0.02, 0.008, 0.002],
        dtype=float,
    )
    response_weight_total = float(np.sum(smooth_response_weights))

    response_rng = np.random.default_rng(73_000 + sample_index)
    coverage = 0.68 + 0.25 * float(response_rng.random())
    one_minus_uniformity = 0.45 + 0.40 * float(response_rng.random())
    phase_purity = 0.70 + 0.25 * float(response_rng.random())
    uniformity_calculated = coverage * one_minus_uniformity * phase_purity
    # Deliberately retain the campaign's warning-only mismatch in sanitized form.
    uniformity_supplied = (
        0.20
        + 0.70
        * float(normalized_physical @ smooth_response_weights[::-1])
        / response_weight_total
    )

    implied_voc = 0.60 + 0.30 * float(response_rng.random())
    optoelectronic = (
        1.50
        + float(normalized_physical @ smooth_response_weights)
        + 0.08 * math.sin(math.pi * normalized_physical[0])
        + 0.04 * normalized_physical[1] * normalized_physical[2]
    )
    photoconductance = (10.0**optoelectronic) / implied_voc

    thickness_target = (
        0.55
        + 0.35
        * float(normalized_physical @ smooth_response_weights)
        / response_weight_total
    )
    thickness_average = 650.0 + 250.0 * math.sqrt(-math.log(thickness_target))
    thickness = math.exp(-(((thickness_average - 650.0) / 250.0) ** 2))
    thickness_values = [
        thickness_average - 6.0,
        thickness_average - 2.0,
        thickness_average + 2.0,
        thickness_average + 6.0,
    ]
    return [
        sample_id,
        *physical,
        coverage,
        1.0 - one_minus_uniformity,
        one_minus_uniformity,
        phase_purity,
        implied_voc,
        photoconductance,
        optoelectronic,
        *thickness_values,
        None,
        thickness_average,
        thickness,
        uniformity_supplied,
        optoelectronic,
        thickness,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]


def write_sanitized_step2c_workbook(
    path: str | Path,
    config: ResolvedStep2CConfig,
    *,
    overwrite: bool = False,
) -> Path:
    """Create a formula-free synthetic v3 workbook with no private recipes."""
    destination = Path(path).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Synthetic Step 2C workbook exists: {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = config.base.workbook_sheet
    worksheet.append(list(SYNTHETIC_STEP2C_HEADERS))
    for sample_index in range(15):
        row = _synthetic_row(config, sample_index)
        if len(row) != len(SYNTHETIC_STEP2C_HEADERS):
            raise RuntimeError("Internal synthetic Step 2C row shape mismatch.")
        worksheet.append(row)
    _, _, expected_max_column, expected_max_row = range_boundaries(
        config.base.expected_content_range
    )
    if expected_max_column != len(SYNTHETIC_STEP2C_HEADERS):
        raise RuntimeError("Synthetic headers do not match the adapter column count.")
    worksheet.cell(row=expected_max_row, column=expected_max_column).value = (
        "SANITIZED SYNTHETIC CI FIXTURE - NO PRIVATE RECIPE"
    )
    workbook.save(destination)
    workbook.close()
    return destination


def _synthetic_config(
    config: ResolvedStep2CConfig,
    *,
    repository_root: Path,
    workbook: Path,
    nested_unique_sizes: Sequence[int],
    anchors_per_selection_step: int,
    omitted_sample_ids: Sequence[int],
    mc_comparison_samples: int,
) -> ResolvedStep2CConfig:
    sizes = tuple(int(value) for value in nested_unique_sizes)
    omissions = tuple(int(value) for value in omitted_sample_ids)
    if len(sizes) != 4 or sizes != tuple(sorted(set(sizes))) or sizes[0] <= 0:
        raise ValueError(
            "Synthetic nested_unique_sizes must be four increasing values."
        )
    if anchors_per_selection_step <= 0 or mc_comparison_samples <= 0:
        raise ValueError("Synthetic anchor and MC counts must be positive.")
    if not omissions or any(
        value not in config.expected_sample_ids for value in omissions
    ):
        raise ValueError(
            "Synthetic omission IDs must be a nonempty subset of configured samples."
        )
    workbook_hash = sha256_file(workbook)
    relative_workbook = workbook.relative_to(repository_root).as_posix()
    fast = ExecutionModeSettings(
        nested_unique_sizes=sizes,
        anchors_per_selection_step=int(anchors_per_selection_step),
        omitted_sample_ids=omissions,
        mc_comparison_samples=int(mc_comparison_samples),
    )
    raw = deepcopy(config.raw)
    raw["workbook"]["path"] = relative_workbook
    raw["workbook"]["expected_sha256"] = workbook_hash.lower()
    synthetic_exception = _synthetic_off_grid_exception(config)
    raw["r0"] = {
        "inherit_from_base": True,
        "synthetic_off_grid_override": {
            "sample_id": synthetic_exception.sample_id,
            "field": synthetic_exception.input_name,
            "value": synthetic_exception.observed_value,
        },
    }
    raw["execution_modes"]["fast"] = {
        "nested_unique_sizes": list(sizes),
        "anchors_per_selection_step": int(anchors_per_selection_step),
        "omitted_sample_ids": list(omissions),
        "mc_comparison_samples": int(mc_comparison_samples),
    }
    base = replace(
        config.base,
        expected_workbook_sha256=workbook_hash,
        off_grid_exceptions=(synthetic_exception,),
    )
    modes = dict(config.execution_modes)
    modes["fast"] = fast
    return replace(
        config,
        raw=raw,
        resolved_config_hash=_canonical_hash(raw),
        base=base,
        workbook_relative_path=relative_workbook,
        workbook_expected_sha256=workbook_hash,
        resolved_off_grid_exceptions=(synthetic_exception,),
        execution_modes=modes,
    )


def run_synthetic_step2c_fast_ci(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    create_portable_zip: bool = True,
    nested_unique_sizes: Sequence[int] = (64, 128, 256, 512),
    anchors_per_selection_step: int = 2,
    omitted_sample_ids: Sequence[int] | None = None,
    mc_comparison_samples: int = 128,
) -> Step2CRobustnessResult:
    """Exercise the full fast orchestration on a sanitized generated workbook.

    The synthetic source and output remain under the ignored Step 2C output root.
    This helper cannot produce a consensus batch because it always runs in fast mode.
    """
    repository_root = Path(__file__).resolve().parents[2]
    config = load_step2c_config(config_path)
    resolved_omissions = (
        (config.expected_sample_ids[0],)
        if omitted_sample_ids is None
        else tuple(omitted_sample_ids)
    )
    destination = Path(output_dir).resolve()
    allowed_root = (repository_root / config.output_root).resolve()
    if allowed_root not in destination.parents:
        raise ValueError(
            "Synthetic Step 2C output must be a child of the configured ignored root."
        )
    source_dir = allowed_root / "synthetic_ci_sources"
    source = source_dir / f"{destination.name}.xlsx"
    write_sanitized_step2c_workbook(source, config, overwrite=overwrite)
    resolved = _synthetic_config(
        config,
        repository_root=repository_root,
        workbook=source,
        nested_unique_sizes=nested_unique_sizes,
        anchors_per_selection_step=anchors_per_selection_step,
        omitted_sample_ids=resolved_omissions,
        mc_comparison_samples=mc_comparison_samples,
    )
    synthetic_config_path = source_dir / f"{destination.name}_resolved_config.yaml"
    if synthetic_config_path.exists() and not overwrite:
        raise FileExistsError(
            f"Synthetic resolved config already exists: {synthetic_config_path}."
        )
    synthetic_config_path.write_text(
        yaml.safe_dump(resolved.raw, sort_keys=False),
        encoding="utf-8",
    )
    resolved = replace(
        resolved,
        config_path=synthetic_config_path.resolve(),
        config_sha256=sha256_file(synthetic_config_path),
    )
    return _run_d2d_step2c_robustness_resolved(
        source,
        resolved,
        destination,
        mode="fast",
        overwrite=overwrite,
        create_portable_zip=create_portable_zip,
        input_data_kind="sanitized_synthetic_ci",
    )


__all__ = [
    "SYNTHETIC_STEP2C_HEADERS",
    "run_synthetic_step2c_fast_ci",
    "write_sanitized_step2c_workbook",
]
