from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from mobo_kit.production_gate import (
    CampaignProposalDisabledError,
    ProductionApprovalError,
    validate_production_config,
)
from mobo_kit.main import run_mobo_experiment


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _valid_config():
    objectives = []
    for name, source, direction in (
        ("uniformity", "APPROVED_UNIFORMITY", "maximize"),
        ("optoelectronic", "APPROVED_OPTO", "maximize"),
        ("thickness", "APPROVED_THICKNESS", "target"),
    ):
        objectives.append(
            {
                "name": name,
                "model_source_column": source,
                "utility_transform": "approved_transform",
                "transform_version": "transform-v1",
                "formula_version": "formula-v1",
                "direction": direction,
                "scaling": {
                    "mode": "already_normalized",
                    "version": "fixed-scale-v1",
                },
            }
        )
    return {
        "schema_version": "d2d-production-v1",
        "campaign_id": "D2D-001",
        "approved_for_production": True,
        "approval": {
            "approved_by": "Scientific and computational owners",
            "approved_at": "2026-07-25T15:00:00-04:00",
            "decision_record": "docs/D2D_MEETING_DECISIONS.md",
        },
        "workbook_profile": "d2d_summary_v2",
        "input_contract_version": "d2d-input-v1",
        "objective_mapping_status": "approved",
        "constraints_status": "approved",
        "objectives": objectives,
        "reference_point_utility": [-0.1, -0.1, -0.1],
        "qc_policy": {
            "complete_case_rule": "all_three_required-v1",
            "failed_measurement_rule": "exclude_with_reason-v1",
            "outlier_rule": "owner_review-v1",
            "control_rule": "declared-control-policy-v1",
            "replicate_rule": "declared-replicate-policy-v1",
        },
        "constraints": [],
        "r1": {
            "method": "ucb_hvi",
            "batch_size": 5,
            "beta": 1.0,
            "posterior_samples": 128,
            "candidate_pool_size": 10000,
        },
        "r2": {
            "method": "qlognehvi",
            "batch_size": 3,
            "mc_samples": 128,
            "candidate_pool_size": 10000,
            "sequential_pending": True,
        },
        "local_penalization": {
            "distance_metric": "normalized_euclidean",
            "radius": 0.2,
            "min_batch_distance": 0.1,
            "min_observed_distance": 0.05,
            "dimension_weights": None,
            "allow_hard_distance_relaxation": False,
        },
        "reproducibility": {
            "seed": 20260725,
            "record_git_commit": True,
            "record_environment_versions": True,
            "record_resolved_config_hash": True,
        },
    }


def test_fully_resolved_config_returns_stable_receipt():
    config = _valid_config()
    first = validate_production_config(config)
    second = validate_production_config(deepcopy(config))
    assert first.objective_count == 3
    assert first.workbook_profile == "d2d_summary_v2"
    assert first.resolved_config_sha256 == second.resolved_config_sha256
    assert len(first.resolved_config_sha256) == 64


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda config: config.pop("approved_for_production"), "explicitly true"),
        (
            lambda config: config.update(approved_for_production=False),
            "explicitly true",
        ),
        (lambda config: config.pop("objectives"), "exactly three"),
        (
            lambda config: config["objectives"][0].pop("formula_version"),
            "formula_version",
        ),
        (
            lambda config: config.pop("reference_point_utility"),
            "reference_point_utility",
        ),
        (lambda config: config.pop("qc_policy"), "qc_policy"),
        (lambda config: config.pop("constraints"), "explicit list"),
        (lambda config: config["r1"].pop("beta"), "r1.beta"),
        (
            lambda config: config["local_penalization"].pop("radius"),
            "local_penalization.radius",
        ),
    ],
)
def test_required_production_fields_fail_closed(mutation, expected):
    config = _valid_config()
    mutation(config)
    with pytest.raises(ProductionApprovalError, match=expected):
        validate_production_config(config)


def test_placeholder_values_are_rejected():
    config = _valid_config()
    config["objectives"][1]["utility_transform"] = "PENDING_AFTER_MEETING"
    config["approval"]["approved_by"] = "TBD"
    with pytest.raises(ProductionApprovalError) as captured:
        validate_production_config(config)
    message = str(captured.value)
    assert "utility_transform" in message
    assert "approval.approved_by" in message

    nested = _valid_config()
    nested["objectives"][0]["scaling"] = {"anchors": {"lower": "TBD"}}
    with pytest.raises(ProductionApprovalError, match="scaling"):
        validate_production_config(nested)


def test_placeholders_anywhere_in_resolved_config_are_rejected_with_paths():
    config = _valid_config()
    config["approval"]["review_metadata"] = {"scientific_review": {"status": "TBD"}}
    config["qc_policy"]["extra_rule"] = "PENDING_AFTER_MEETING"
    with pytest.raises(ProductionApprovalError) as captured:
        validate_production_config(config)
    message = str(captured.value)
    assert "approval.review_metadata.scientific_review.status" in message
    assert "qc_policy.extra_rule" in message


def test_scaling_must_be_fixed_and_cannot_reference_observed_data():
    dynamic = _valid_config()
    dynamic["objectives"][0]["scaling"] = {
        "mode": "observed_minmax",
        "version": "round-specific-v1",
    }
    with pytest.raises(ProductionApprovalError, match="data-derived scaling"):
        validate_production_config(dynamic)

    hidden_dynamic = _valid_config()
    hidden_dynamic["objectives"][0]["scaling"]["anchor_source"] = "observed_data"
    with pytest.raises(ProductionApprovalError, match="unsupported field"):
        validate_production_config(hidden_dynamic)

    fixed = _valid_config()
    fixed["objectives"][0]["scaling"] = {
        "mode": "fixed_affine",
        "version": "fixed-negative-anchor-v1",
        "lower_anchor": -2.0,
        "upper_anchor": 3.0,
    }
    assert validate_production_config(fixed).objective_count == 3


@pytest.mark.parametrize(
    "path, value",
    [
        (("r1", "batch_size"), 5.0),
        (("r1", "posterior_samples"), 12.5),
        (("r1", "candidate_pool_size"), 100.25),
        (("r2", "batch_size"), 3.0),
        (("r2", "mc_samples"), 16.5),
        (("r2", "candidate_pool_size"), 200.5),
    ],
)
def test_sample_pool_and_batch_counts_require_integer_types(path, value):
    config = _valid_config()
    config[path[0]][path[1]] = value
    with pytest.raises(ProductionApprovalError, match=path[1]):
        validate_production_config(config)


def test_distance_weights_are_explicit_and_metric_consistent():
    ordinary = _valid_config()
    ordinary["local_penalization"]["dimension_weights"] = [1.0]
    with pytest.raises(ProductionApprovalError, match="requires dimension_weights"):
        validate_production_config(ordinary)

    weighted = _valid_config()
    weighted["inputs"] = [{"name": "x1"}, {"name": "x2"}]
    weighted["local_penalization"]["distance_metric"] = "weighted_normalized_euclidean"
    weighted["local_penalization"]["dimension_weights"] = [1.0, 0.0]
    with pytest.raises(ProductionApprovalError, match="finite positive"):
        validate_production_config(weighted)
    weighted["local_penalization"]["dimension_weights"] = [1.0, 2.0]
    assert validate_production_config(weighted).objective_count == 3


def test_nonfinite_extra_configuration_cannot_receive_a_receipt():
    config = _valid_config()
    config["approval"]["extra_numeric_provenance"] = float("nan")
    with pytest.raises(ProductionApprovalError, match="JSON-serializable"):
        validate_production_config(config)


def test_supplied_provisional_template_cannot_pass_gate():
    path = REPOSITORY_ROOT / "configs" / "d2d_step2a_provisional.yaml"
    with path.open("r", encoding="utf-8") as stream:
        provisional = yaml.safe_load(stream)
    assert provisional["approved_for_production"] is False
    with pytest.raises(ProductionApprovalError) as captured:
        validate_production_config(provisional)
    assert "explicitly true" in str(captured.value)
    assert len(captured.value.errors) >= 10


def test_campaign_runner_blocks_provisional_config_before_csv_or_output(tmp_path):
    output_dir = tmp_path / "must_not_exist"
    provisional_path = REPOSITORY_ROOT / "configs" / "d2d_step2a_provisional.yaml"
    with pytest.raises(ProductionApprovalError):
        run_mobo_experiment(
            csv_path=str(tmp_path / "missing.csv"),
            save_dir=str(output_dir),
            config_path=str(provisional_path),
            device="cpu",
            verbose=False,
            propose_candidates=True,
        )
    assert not output_dir.exists()


def test_campaign_runner_blocks_even_approved_config_from_legacy_path(tmp_path):
    config_path = tmp_path / "approved.yaml"
    config_path.write_text(yaml.safe_dump(_valid_config()), encoding="utf-8")
    output_dir = tmp_path / "must_not_exist"
    with pytest.raises(CampaignProposalDisabledError, match="Step 2B campaign adapter"):
        run_mobo_experiment(
            csv_path=str(tmp_path / "missing.csv"),
            save_dir=str(output_dir),
            config_path=str(config_path),
            device="cpu",
            verbose=False,
            propose_candidates=True,
        )
    assert not output_dir.exists()


def test_campaign_runner_rejects_auto_config_for_proposals_before_csv_parse(tmp_path):
    output_dir = tmp_path / "must_not_exist"
    with pytest.raises(ProductionApprovalError, match="explicit resolved"):
        run_mobo_experiment(
            csv_path=str(tmp_path / "missing.csv"),
            save_dir=str(output_dir),
            config_path=None,
            device="cpu",
            verbose=False,
            propose_candidates=True,
        )
    assert not output_dir.exists()
