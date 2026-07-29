from pathlib import Path

import pytest
import yaml

from mobo_kit.d2d_campaign import D2D_DEBUG_WATERMARK, D2D_INPUT_COLUMNS
from mobo_kit.d2d_step2c_config import (
    STEP2C_NESTED_POOL_SIZES,
    STEP2C_OUTPUT_ROOT,
    STEP2C_RUNTIME_GENERATED,
    STEP2C_SYNTHETIC_SOURCE_KIND,
    Step2CConfigError,
    load_step2c_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "d2d_step2c_debug.yaml"


def _copy_with_change(tmp_path: Path, mutator) -> Path:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    mutator(raw)
    path = tmp_path / "changed_step2c.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_step2c_config_resolves_public_safe_debug_contract() -> None:
    config = load_step2c_config(CONFIG_PATH)

    assert config.config_path == CONFIG_PATH.resolve()
    assert len(config.config_sha256) == 64
    assert len(config.resolved_config_hash) == 64
    assert tuple(config.design.names) == D2D_INPUT_COLUMNS
    assert len(config.expected_sample_ids) == 15
    assert config.workbook_source_kind == STEP2C_SYNTHETIC_SOURCE_KIND
    assert config.workbook_relative_path == STEP2C_RUNTIME_GENERATED
    assert config.workbook_expected_sha256 == STEP2C_RUNTIME_GENERATED.upper()
    assert config.raw["workbook"] == {"source_kind": STEP2C_SYNTHETIC_SOURCE_KIND}
    assert config.raw["r0"] == {"inherit_from_base": True}
    assert config.nested_pool_sizes == STEP2C_NESTED_POOL_SIZES
    assert config.beta_values == (1.0, 4.0, 9.0)
    assert config.bound_policies == ("none", "clip_ucb")
    assert config.primary_bound_policy == "clip_ucb"
    assert config.primary_penalty_variant == "radius_0_25"
    assert config.model_variant_names == ("dim_scaled_prior", "conservative")
    assert config.influence_sample_ids == config.expected_sample_ids
    assert config.shortlist_min == 8
    assert config.shortlist_max == 12
    assert config.output_root == STEP2C_OUTPUT_ROOT
    assert config.raw["debug_watermark"] == D2D_DEBUG_WATERMARK
    assert config.raw["approved_for_experiment"] is False
    assert config.raw["approved_for_production"] is False
    assert config.mode("full").nested_unique_sizes == STEP2C_NESTED_POOL_SIZES
    assert config.mode("fast").omitted_sample_ids == config.expected_sample_ids[:3]


@pytest.mark.parametrize(
    "mutator,match",
    [
        (
            lambda raw: raw.__setitem__("approved_for_experiment", True),
            "approved_for_experiment",
        ),
        (
            lambda raw: raw["workbook"].__setitem__(
                "objective_columns", ["Y", "Z", "AA"]
            ),
            "workbook",
        ),
        (
            lambda raw: raw["objectives"].__setitem__("reference_point", [0, 0, 0]),
            "reference_point",
        ),
        (
            lambda raw: raw["r0"].__setitem__(
                "include_control_in_primary_model", False
            ),
            "r0",
        ),
        (
            lambda raw: raw["candidate_search"].__setitem__(
                "preserve_accepted_prefix_nesting", False
            ),
            "nesting",
        ),
        (
            lambda raw: raw["local_penalty_study"].__setitem__(
                "hard_distance_relaxation", True
            ),
            "relaxation",
        ),
        (
            lambda raw: raw["ucb_hvi"].__setitem__("primary_beta", 9.0),
            "primary UCB-HVI",
        ),
        (
            lambda raw: raw["ucb_hvi"].__setitem__("primary_bound_policy", "none"),
            "clipping",
        ),
        (
            lambda raw: raw["observation_influence"].__setitem__(
                "common_pool_size", 65536
            ),
            "common pool",
        ),
        (
            lambda raw: raw["robust_regions"].__setitem__(
                "primary_distance_threshold", 0.20
            ),
            "Robust-region",
        ),
        (
            lambda raw: raw["execution_modes"]["fast"].__setitem__(
                "anchors_per_selection_step", 4
            ),
            "execution_modes",
        ),
        (
            lambda raw: raw["outputs"].__setitem__("tracked_private_recipes", True),
            "outputs",
        ),
    ],
)
def test_step2c_config_fails_closed_on_safety_and_scientific_changes(
    tmp_path: Path, mutator, match: str
) -> None:
    changed = _copy_with_change(tmp_path, mutator)
    with pytest.raises(Step2CConfigError, match=match):
        load_step2c_config(changed)


def test_step2c_config_rejects_unknown_execution_mode() -> None:
    config = load_step2c_config(CONFIG_PATH)
    with pytest.raises(Step2CConfigError, match="fast.*full"):
        config.mode("production")


def test_step2c_config_accepts_runtime_synthetic_identity(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["workbook"] = {
        "source_kind": STEP2C_SYNTHETIC_SOURCE_KIND,
        "path": (
            "local_outputs/d2d_step2c_robustness/"
            "synthetic_ci_sources/public_fixture.xlsx"
        ),
        "expected_sha256": "a" * 64,
    }
    path = tmp_path / "resolved_synthetic_step2c.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_step2c_config(path)

    assert config.workbook_source_kind == STEP2C_SYNTHETIC_SOURCE_KIND
    assert config.workbook_expected_sha256 == "A" * 64
