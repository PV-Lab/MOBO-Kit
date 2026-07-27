from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mobo_kit.d2d_scores import compute_thickness_average, compute_thickness_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    REPOSITORY_ROOT
    / "notebooks"
    / "D2D_MOBO_TEST Global Distance Candidate generation.ipynb"
)


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def test_d2d_notebook_contains_guarded_score_section_in_order() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    sources = [_source(cell) for cell in cells]
    section_2 = next(
        index
        for index, text in enumerate(sources)
        if text.startswith("## 2. Normalize Data")
    )
    section_25 = next(
        index
        for index, text in enumerate(sources)
        if text.startswith("## 2.5 Calculate and Validate D2D Scores")
    )
    section_3 = next(
        index
        for index, text in enumerate(sources)
        if text.startswith("## 3. Fit Gaussian Process")
    )
    assert section_2 < section_25 < section_3

    section_text = "\n".join(sources[section_25:section_3])
    assert "exp(-((mean_t - 650.0) / 250.0) ** 2)" in section_text
    assert '"Uniformity score"' in section_text
    assert '"Optoelectronic score"' in section_text
    assert '"Thickness score"' in section_text
    assert "T1" in section_text and "T4" in section_text
    assert "T anom" in section_text
    assert "validate_supplied_d2d_scores" in section_text
    assert "from mobo_kit.d2d_scores import" in section_text

    section_code = "\n".join(
        _source(cell)
        for cell in cells[section_25:section_3]
        if cell["cell_type"] == "code"
    )
    assert "-0.5" not in section_code
    assert "0.5 *" not in section_code


def test_d2d_notebook_is_portable_cleared_and_uses_active_apis() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    all_source = "\n".join(_source(cell) for cell in notebook["cells"])
    assert "/Users/" not in all_source
    assert "C:\\Users\\" not in all_source
    for retired_name in (
        "MixedMCMultiOutputObjective",
        "fit_gp_models_baybe",
        "propose_batch_discrete",
        "random_discrete_choices",
    ):
        assert retired_name not in all_source
    assert "run_d2d_step2b_debug" in all_source
    assert "DEBUG ONLY - NOT APPROVED FOR EXPERIMENT" in all_source
    assert "diagnostics_path.mkdir(parents=True, exist_ok=True)" in all_source
    assert "save_path.mkdir" not in all_source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []


def test_d2d_notebook_narrative_enforces_resolved_step2b_contract() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    markdown_source = "\n".join(
        _source(cell) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    all_source = "\n".join(_source(cell) for cell in notebook["cells"])

    for required_contract_text in (
        "Excel Z `Uniformity score`",
        "AA `Optoelectronic score`",
        "AB `Thickness score`",
        "R1 UCB-HVI",
        "**max/max/max**",
        "`[-0.01, -10.0, -0.01]`",
        "DEBUG ONLY - NOT APPROVED FOR EXPERIMENT",
    ):
        assert required_contract_text in markdown_source

    for contradictory_legacy_text in (
        "PCE, Stability, Repeatability",
        'out["X_phys"]',
        'out["X_norm"]',
        'out["acq_val"]',
        "use_lognehvi",
        "log nEHVI",
        "Before running the batch in the lab",
        "hand off to execution",
        "slightly worse than the worst feasible values",
        "lab_worklist",
    ):
        assert contradictory_legacy_text not in all_source


def test_notebook_thickness_helper_fixture_uses_no_half_factor() -> None:
    expected = pytest.approx(0.9607894391523232)
    assert compute_thickness_score([600.0, 600.0, None, "\u00a0"]) == expected


def test_notebook_thickness_calculation_cell_executes_against_package_api() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cell_source = next(
        _source(cell)
        for cell in notebook["cells"]
        if "thickness_means = normalized_thickness.apply" in _source(cell)
    )
    normalized_thickness = pd.DataFrame(
        [[600.0, 600.0, None, "\u00a0"]], columns=["T1", "T2", "T3", "T4"]
    )
    namespace = {
        "compute_thickness_average": compute_thickness_average,
        "compute_thickness_score": compute_thickness_score,
        "data_rows": pd.DataFrame(
            {"Sample number": [1], "Thickness score": [0.9607894391523232]}
        ),
        "display": lambda _value: None,
        "normalized_thickness": normalized_thickness,
        "pd": pd,
    }

    exec(compile(cell_source, str(NOTEBOOK_PATH), "exec"), namespace)

    assert namespace["thickness_means"].iloc[0] == pytest.approx(600.0)
    assert namespace["calculated_thickness_scores"].iloc[0] == pytest.approx(
        0.9607894391523232
    )
