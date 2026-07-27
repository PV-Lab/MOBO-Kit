from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import mobo_kit.d2d_step2b_debug as debug_module
from mobo_kit.d2d_campaign import D2D_DEBUG_WATERMARK, D2D_INPUT_COLUMNS
from mobo_kit.d2d_step2b_debug import run_d2d_step2b_debug


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CONFIG = REPOSITORY_ROOT / "configs" / "d2d_step2b_debug.yaml"
PRIVATE_WORKBOOK = os.environ.get("MOBO_KIT_D2D_PRIVATE_WORKBOOK")
PRIVATE_CONFIG = os.environ.get("MOBO_KIT_D2D_PRIVATE_CONFIG")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _materialized_public_config(tmp_path: Path, expected_sha256: str) -> Path:
    config = yaml.safe_load(PUBLIC_CONFIG.read_text(encoding="utf-8"))
    config["template_only"] = False
    config["workbook"]["expected_sha256"] = expected_sha256
    path = tmp_path / "materialized-public-debug.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_debug_adapter_creates_no_output_before_workbook_hash_validation(
    tmp_path: Path,
) -> None:
    config_path = _materialized_public_config(tmp_path, "0" * 64)
    workbook = tmp_path / "source" / "not_the_campaign.xlsx"
    workbook.parent.mkdir()
    workbook.write_bytes(b"not an xlsx")
    output = tmp_path / "must_not_exist"

    with pytest.raises(ValueError, match="Workbook hash does not match"):
        run_d2d_step2b_debug(workbook, config_path, output, run_sensitivity=False)
    assert not output.exists()


def test_debug_adapter_rejects_output_inside_source_workbook_directory(
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "source.xlsx"
    workbook.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="source workbook or its directory"):
        run_d2d_step2b_debug(
            workbook,
            PUBLIC_CONFIG,
            tmp_path / "debug_output",
            run_sensitivity=False,
        )


def test_debug_adapter_rejects_repository_docs_output_before_writing(
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "source.xlsx"
    workbook.write_bytes(b"placeholder")
    copied_config = _materialized_public_config(tmp_path, "0" * 64)
    forbidden = REPOSITORY_ROOT / "docs" / "step2b-forbidden-test-output"
    assert not forbidden.exists()

    with pytest.raises(ValueError, match="configured debug output root"):
        run_d2d_step2b_debug(
            workbook,
            copied_config,
            forbidden,
            run_sensitivity=False,
        )
    assert not forbidden.exists()


def test_compute_proposal_uses_independent_pool_and_mc_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, int] = {}
    dimensions = len(D2D_INPUT_COLUMNS)
    pool = SimpleNamespace(
        X_norm=np.zeros((5, dimensions)),
        X_phys=np.zeros((5, dimensions)),
    )
    selection = SimpleNamespace(X_phys=np.zeros((5, dimensions)))

    def fake_pool(*args: object, seed: int, **kwargs: object) -> object:
        captured["pool_seed"] = seed
        return pool

    def fake_proposal(*args: object, seed: int, **kwargs: object) -> object:
        captured["mc_seed"] = seed
        return SimpleNamespace(selection=selection)

    monkeypatch.setattr(debug_module, "sample_discrete_candidate_pool", fake_pool)
    monkeypatch.setattr(debug_module, "propose_ucb_hvi_batch", fake_proposal)
    config = SimpleNamespace(
        design=object(),
        min_observed_distance=0.0,
        dimension_weights=None,
        reference_point_utility=np.array([-0.01, -10.0, -0.01]),
        r1_batch_size=5,
        score_chunk_size=32,
    )
    training = SimpleNamespace(
        X_phys_all=np.zeros((1, dimensions)),
        on_grid_mask=np.ones(1, dtype=bool),
        X_norm_all=np.zeros((1, dimensions)),
    )

    debug_module._compute_proposal(
        config,
        training,
        object(),
        torch.zeros((1, dimensions), dtype=torch.double),
        torch.zeros((1, 3), dtype=torch.double),
        pool_size=5,
        pool_seed=137,
        mc_seed=73,
        beta=4.0,
        posterior_samples=16,
        radius=0.25,
        min_batch_distance=0.15,
    )

    assert captured == {"pool_seed": 137, "mc_seed": 73}


@pytest.mark.skipif(
    not PRIVATE_WORKBOOK or not PRIVATE_CONFIG,
    reason=(
        "set MOBO_KIT_D2D_PRIVATE_WORKBOOK and MOBO_KIT_D2D_PRIVATE_CONFIG "
        "to opt into the ignored private integration test"
    ),
)
def test_local_debug_adapter_is_deterministic_complete_and_read_only(
    tmp_path: Path,
) -> None:
    local_workbook = Path(str(PRIVATE_WORKBOOK)).expanduser().resolve()
    private_config = Path(str(PRIVATE_CONFIG)).expanduser().resolve()
    private_raw = yaml.safe_load(private_config.read_text(encoding="utf-8"))
    before_hash = _sha256(local_workbook)
    before_mtime = local_workbook.stat().st_mtime_ns
    assert before_hash == str(private_raw["workbook"]["expected_sha256"]).upper()

    first = run_d2d_step2b_debug(
        local_workbook,
        private_config,
        tmp_path / "run_1",
        run_sensitivity=False,
    )
    second = run_d2d_step2b_debug(
        local_workbook,
        private_config,
        tmp_path / "run_2",
        run_sensitivity=False,
    )

    assert first.candidates_unique.shape[0] == 5
    assert first.replicate_worklist.shape[0] == 15
    assert first.candidates_unique["candidate_id"].tolist() == [
        "R1-C01",
        "R1-C02",
        "R1-C03",
        "R1-C04",
        "R1-C05",
    ]
    assert first.candidates_unique["debug_only"].all()
    assert not first.candidates_unique["approved_for_experiment"].any()
    assert first.candidates_unique["candidate_status"].eq(D2D_DEBUG_WATERMARK).all()
    assert first.candidates_unique["known_uniformity_score_mismatch"].all()
    assert {
        "campaign_id",
        "round",
        "sample_id",
        "candidate_id",
        "row_role",
        "replicate_group",
        "replicate_number",
        "candidate_status",
        "include_in_model",
        "measurement_provenance",
        "off_grid_exception",
        "exclusion_reason",
    } <= set(first.candidates_unique.columns)
    assert first.replicate_worklist["known_uniformity_score_mismatch"].all()
    assert first.candidates_unique["grid_valid"].all()
    assert first.candidates_unique["bounds_valid"].all()
    assert (first.candidates_unique["base_ucb_hvi"] > 0).all()
    assert (
        not first.candidates_unique.loc[:, list(D2D_INPUT_COLUMNS)].duplicated().any()
    )
    np.testing.assert_array_equal(
        first.candidates_unique.loc[:, list(D2D_INPUT_COLUMNS)].to_numpy(),
        second.candidates_unique.loc[:, list(D2D_INPUT_COLUMNS)].to_numpy(),
    )
    assert first.run_manifest["training_row_count"] == 15
    assert first.run_manifest["uniformity_warning_count"] == 11
    assert first.run_manifest["reference_point_utility"] == [-0.01, -10.0, -0.01]
    assert first.run_manifest["objective_order"] == [
        "Uniformity score",
        "Optoelectronic score",
        "Thickness score",
    ]
    baseline_sensitivity = first.sensitivity_summary.iloc[0]
    assert baseline_sensitivity["exact_overlap_with_baseline"] == 5
    assert baseline_sensitivity["jaccard_overlap_with_baseline"] == 1.0
    assert baseline_sensitivity["mean_nearest_batch_distance_to_baseline"] == 0.0
    assert baseline_sensitivity["pool_seed"] == 73
    assert baseline_sensitivity["mc_seed"] == 73
    assert bool(baseline_sensitivity["debug_only"])
    assert not bool(baseline_sensitivity["approved_for_experiment"])
    assert baseline_sensitivity["candidate_status"] == D2D_DEBUG_WATERMARK
    assert bool(baseline_sensitivity["known_uniformity_score_mismatch"])
    assert "boundary_coordinate_count" in first.sensitivity_summary.columns
    assert "boundary_dimension_count" not in first.sensitivity_summary.columns
    assert "ordered_batch_sha256" in first.sensitivity_summary.columns
    assert "candidate_set_sha256" not in first.sensitivity_summary.columns
    assert baseline_sensitivity["observed_pareto_count"] == 6
    assert json.loads(baseline_sensitivity["observed_pareto_sample_ids"])

    sensitivity_candidates = first.sensitivity_candidates
    assert sensitivity_candidates.shape[0] == 5
    assert set(sensitivity_candidates["record_type"]) == {"selected_candidate"}
    assert set(sensitivity_candidates["run_label"]) == {"baseline"}
    assert set(sensitivity_candidates["pool_seed"]) == {73}
    assert set(sensitivity_candidates["mc_seed"]) == {73}
    assert set(sensitivity_candidates["candidate_status"]) == {D2D_DEBUG_WATERMARK}
    assert sensitivity_candidates["known_uniformity_score_mismatch"].all()
    assert {
        *D2D_INPUT_COLUMNS,
        "base_raw_score",
        "base_log_score",
        "final_penalized_score",
        "penalized_log_score",
        "penalty_factor",
        "pairwise_distance_row",
        "nearest_observed_distance",
        "boundary_coordinates",
        "boundary_coordinate_count",
        "total_fit_proposal_runtime_seconds",
        "status",
        "warning",
    } <= set(sensitivity_candidates.columns)
    np.testing.assert_allclose(
        sensitivity_candidates["final_penalized_score"],
        sensitivity_candidates["base_raw_score"]
        * sensitivity_candidates["penalty_factor"],
    )
    assert all(
        len(json.loads(value)) == 5
        for value in sensitivity_candidates["pairwise_distance_row"]
    )
    assert first.control_ablation.empty
    assert {
        "baseline_predicted_uniformity_score_mean",
        "control_excluded_predicted_uniformity_score_mean",
        "delta_uniformity_score_mean",
        "selected_exact_overlap_count",
        "total_fit_proposal_runtime_seconds",
        "status",
        "candidate_status",
        "known_uniformity_score_mismatch",
    } <= set(first.control_ablation.columns)

    required = {
        "DEBUG_ONLY_NOT_APPROVED_FOR_EXPERIMENT.txt",
        "workbook_audit.json",
        "score_validation.csv",
        "training_row_manifest.csv",
        "model_diagnostics.csv",
        "r1_debug_candidates_unique.csv",
        "r1_debug_replicate_worklist.csv",
        "candidate_diagnostics.csv",
        "sensitivity_summary.csv",
        "sensitivity_candidates_long.csv",
        "control_ablation.csv",
        "run_manifest.json",
    }
    assert required <= {path.name for path in first.output_dir.iterdir()}
    plot_paths = list((first.output_dir / "plots").glob("*.png"))
    assert len(plot_paths) == 4
    assert all(D2D_DEBUG_WATERMARK.encode() in path.read_bytes() for path in plot_paths)
    written_candidates = pd.read_csv(
        first.output_dir / "sensitivity_candidates_long.csv"
    )
    assert written_candidates.shape[0] == 5
    written_control = pd.read_csv(first.output_dir / "control_ablation.csv")
    assert written_control.empty
    written_diagnostics = pd.read_csv(first.output_dir / "candidate_diagnostics.csv")
    assert written_diagnostics["candidate_status"].eq(D2D_DEBUG_WATERMARK).all()
    assert written_diagnostics["known_uniformity_score_mismatch"].all()
    written_training = pd.read_csv(first.output_dir / "training_row_manifest.csv")
    assert written_training["candidate_status"].eq(D2D_DEBUG_WATERMARK).all()
    assert written_training["known_uniformity_score_mismatch"].all()
    assert _sha256(local_workbook) == before_hash
    assert local_workbook.stat().st_mtime_ns == before_mtime
