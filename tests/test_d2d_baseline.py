from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
import botorch
import gpytorch

import mobo_kit
from mobo_kit.cli import main as cli_main
from mobo_kit.constraints import apply_row_constraints, constraints_from_config
from mobo_kit.design import InputSpec, build_design, build_design_from_config
from mobo_kit.lhs import lhs_dataframe_optimized
from mobo_kit.main import (
    _validate_candidate_batch,
    generate_initial_experiments,
    run_mobo_experiment,
)
from mobo_kit.utils import select_device


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_D2D_CONFIG = REPOSITORY_ROOT / "configs" / "FA0.9CS0.1PbI3_260407_Config.yaml"

EXPECTED_INPUTS = [
    ("speed_1", "rpm", 1000.0, 6000.0, 500.0, 11),
    ("time_1", "s", 5.0, 50.0, 5.0, 10),
    ("speed_2", "rpm", 0.0, 5000.0, 500.0, 11),
    ("time_2", "s", 10.0, 60.0, 5.0, 11),
    ("precur_conc", "M", 1.0, 2.0, 0.05, 21),
    ("precur_vol", "uL", 40.0, 200.0, 10.0, 17),
    ("anneal_temp", "C", 100.0, 185.0, 5.0, 18),
    ("anneal_time", "min", 10.0, 60.0, 5.0, 11),
    ("anti_vol", "uL", 100.0, 200.0, 5.0, 21),
    ("anti_time", "s", 9.0, 25.0, 2.0, 9),
]


def _load_canonical_config() -> dict:
    with CANONICAL_D2D_CONFIG.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _assert_exact_grid_membership(frame: pd.DataFrame, design) -> None:
    for column, grid in zip(design.names, design.var_list):
        assert np.all(np.isin(frame[column].to_numpy(), grid))


def test_canonical_d2d_yaml_has_exact_input_contract():
    config = _load_canonical_config()
    design = build_design_from_config(config)

    assert config["campaign"]["name"] == "D2D_FA0.9Cs0.1PbI3"
    # the config is no longer a baseline-only stub; it now carries a resolved
    # objective contract and can propose candidates
    assert config["campaign"]["status"] == "active"
    assert config["objectives"]["contract_version"]
    assert len(config["objectives"]["specs"]) == 3
    assert config["constraints"] == []
    assert design.names == [item[0] for item in EXPECTED_INPUTS]
    assert design.units == [item[1] for item in EXPECTED_INPUTS]
    assert np.array_equal(
        design.lowers,
        np.asarray([item[2] for item in EXPECTED_INPUTS]),
    )
    assert np.array_equal(
        design.uppers,
        np.asarray([item[3] for item in EXPECTED_INPUTS]),
    )
    assert np.array_equal(
        design.steps,
        np.asarray([item[4] for item in EXPECTED_INPUTS]),
    )


def test_canonical_d2d_grids_have_exact_membership_and_cardinality():
    design = build_design_from_config(_load_canonical_config())

    for grid, expected in zip(design.var_list, EXPECTED_INPUTS):
        _, _, start, stop, step, cardinality = expected
        expected_grid = np.round(
            start + step * np.arange(cardinality, dtype=float),
            6,
        )
        assert len(grid) == cardinality
        assert np.array_equal(grid, expected_grid)
        assert grid[0] == start
        assert grid[-1] == stop

    assert int(np.prod([len(grid) for grid in design.var_list], dtype=np.int64)) == (
        177_816_994_740
    )


def test_canonical_d2d_lhs_is_deterministic_unique_and_ten_dimensional():
    design = build_design_from_config(_load_canonical_config())
    kwargs = {
        "design": design,
        "n": 20,
        "seed": 42,
        "max_abs_corr": 0.32,
        "max_attempts": 10,
        "samples_per_attempt": 100,
        "subset_tries": 2000,
    }

    first = lhs_dataframe_optimized(**kwargs)
    second = lhs_dataframe_optimized(**kwargs)

    pd.testing.assert_frame_equal(first, second)
    assert first.shape == (20, 10)
    assert list(first.columns) == design.names
    assert not first.duplicated().any()
    _assert_exact_grid_membership(first, design)

    correlations = np.abs(np.corrcoef(first.to_numpy(), rowvar=False))
    np.fill_diagonal(correlations, 0.0)
    assert float(np.max(correlations)) <= 0.32


def test_canonical_constraints_default_to_empty():
    config = _load_canonical_config()
    design = build_design_from_config(config)

    assert constraints_from_config(config, design) == []


def test_one_explicit_supported_constraint_builds_and_applies():
    config = {
        "inputs": [
            {
                "name": "absolute_humidity",
                "unit": "g/m^3",
                "start": 0,
                "stop": 30,
                "step": 1,
            },
            {
                "name": "temperature_c",
                "unit": "C",
                "start": 20,
                "stop": 30,
                "step": 1,
            },
        ],
        "constraints": [
            {
                "clausius_clapeyron": True,
                "ah_col": "absolute_humidity",
                "temp_c_col": "temperature_c",
            }
        ],
    }
    design = build_design_from_config(config)
    constraints = constraints_from_config(config, design)
    rows = np.asarray([[10.0, 20.0], [30.0, 20.0]])

    assert len(constraints) == 1
    assert np.array_equal(
        apply_row_constraints(rows, design, constraints),
        np.asarray([True, False]),
    )


@pytest.mark.parametrize(
    ("ah_col", "temp_col", "missing"),
    [
        ("absolute_humidity", "anneal_temp", "absolute_humidity"),
        ("anti_vol", "temperature_c", "temperature_c"),
    ],
)
def test_constraint_references_to_missing_columns_fail_clearly(
    ah_col: str,
    temp_col: str,
    missing: str,
):
    config = _load_canonical_config()
    config["constraints"] = [
        {
            "clausius_clapeyron": True,
            "ah_col": ah_col,
            "temp_c_col": temp_col,
        }
    ]
    design = build_design_from_config(config)

    with pytest.raises(KeyError, match=re.escape(f"column '{missing}'")):
        constraints_from_config(config, design)


@pytest.mark.parametrize(
    ("arguments", "expected_tokens"),
    [
        (["--help"], ["generate", "run"]),
        (
            ["generate", "--help"],
            ["--config", "--n-samples", "--out", "--max-corr"],
        ),
        (
            ["run", "--help"],
            ["--csv", "--device", "--propose-candidates", "--reference-point"],
        ),
    ],
)
def test_cli_help_surfaces_are_importable(
    arguments,
    expected_tokens,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(sys, "argv", ["mobo-kit", *arguments])
    with pytest.raises(SystemExit) as exit_info:
        cli_main()
    captured = capsys.readouterr()
    output = captured.out + captured.err

    assert exit_info.value.code == 0
    assert "Traceback" not in output
    for token in expected_tokens:
        assert token in output


def test_package_import_and_cpu_smoke():
    assert Path(mobo_kit.__file__).resolve().is_file()
    assert torch.__version__
    assert gpytorch.__version__
    assert botorch.__version__
    cpu = select_device("cpu")
    tensor = torch.tensor([1.0, 2.0], dtype=torch.float64, device=cpu)

    assert cpu.type == "cpu"
    assert tensor.device.type == "cpu"
    assert tensor.sum().item() == 3.0


def test_production_source_and_notebooks_have_no_personal_absolute_paths():
    forbidden_fragments = (
        "/Users/",
        "C:\\Users\\",
        "C:\\\\Users\\\\",
        "Dropbox/Buonassisi-Group",
        "Dropbox\\Buonassisi-Group",
        "Dropbox\\\\Buonassisi-Group",
    )
    inspected_paths = [
        *sorted((REPOSITORY_ROOT / "src").rglob("*.py")),
        *sorted((REPOSITORY_ROOT / "notebooks").rglob("*.ipynb")),
    ]

    findings = []
    for path in inspected_paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for fragment in forbidden_fragments:
            if fragment in text:
                findings.append(f"{path.relative_to(REPOSITORY_ROOT)}: {fragment}")

    assert findings == []


def test_candidate_batch_validation_fails_closed():
    design = build_design([InputSpec("x", 0, 1, 1), InputSpec("y", 0, 1, 1)])
    observed = pd.DataFrame([[0.0, 0.0]], columns=design.names)

    valid = _validate_candidate_batch(
        {"X_phys": [[0.0, 1.0], [1.0, 0.0]]},
        design,
        observed,
        batch_size=2,
    )
    np.testing.assert_array_equal(valid, [[0.0, 1.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="expected shape"):
        _validate_candidate_batch(
            {"X_phys": [[0.0, 1.0]]}, design, observed, batch_size=2
        )
    with pytest.raises(ValueError, match="duplicate snapped recipes"):
        _validate_candidate_batch(
            {"X_phys": [[0.0, 1.0], [0.0, 1.0]]},
            design,
            observed,
            batch_size=2,
        )
    with pytest.raises(ValueError, match="repeats observed recipes"):
        _validate_candidate_batch(
            {"X_phys": [[0.0, 0.0], [1.0, 1.0]]},
            design,
            observed,
            batch_size=2,
        )
    with pytest.raises(ValueError, match="off-grid"):
        _validate_candidate_batch(
            {"X_phys": [[0.0, 0.5], [1.0, 1.0]]},
            design,
            observed,
            batch_size=2,
        )


def test_explicit_missing_config_path_fails_instead_of_inference(tmp_path):
    missing_config = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        run_mobo_experiment(
            csv_path="not-read-before-config-validation.csv",
            config_path=str(missing_config),
            verbose=False,
            device="cpu",
        )


def test_generate_baseline_is_safe_and_does_not_propose_candidates(tmp_path):
    output_path = tmp_path / "nested" / "d2d_round_0.csv"

    result = generate_initial_experiments(
        config_path=str(CANONICAL_D2D_CONFIG),
        n_samples=20,
        save_path=str(output_path),
        seed=42,
        verbose=False,
        max_abs_corr=0.32,
        max_attempts=10,
    )
    generated = pd.read_csv(output_path)
    design = build_design_from_config(_load_canonical_config())

    assert result["status"] == "success"
    assert result["n_samples"] == 20
    assert result["variables"] == design.names
    assert result["constraints_applied"] is False
    assert generated.shape == (20, 10)
    assert list(generated.columns) == design.names
    assert not generated.duplicated().any()
    _assert_exact_grid_membership(generated, design)
