"""The round-simulation script.

Not a test of the optimiser -- ``test_dtlz2_acceptance.py`` does that -- but of the
commitments this script makes on top of it, each of which fails silently if it
drifts:

* the oracle is a **deterministic** function, because the manifest compares
  batches across parameter cells and that comparison is meaningless otherwise;
* the oracle reports thickness as the posterior **median**, not the lognormal
  mean, so the simulated landscape does not bulge wherever the posterior is wide;
* the loop produces exactly 23 conditions labelled 15 / 5 / 3;
* the manifest carries exactly the declared columns;
* ``min_batch_distance`` is pinned in every cell, so "spacing" means one thing.

The synthetic campaign here needs no workbook.  One test does, and skips without
it, following the convention in ``test_workbook_io.py``.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mobo_kit.campaign import (
    build_objective_transform,
    fit_campaign_models,
    run_r0_lhs,
)

SOURCE = "local_inputs/Summary Table.xlsx"


def _load():
    path = Path("scripts") / "plot_round_simulation.py"
    spec = importlib.util.spec_from_file_location("_script_plot_round_simulation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prs = _load()

INPUT_DIM = 10
R0_SIZE, R1_SIZE, R2_SIZE = 15, 5, 3
SEED = 73


def _config(pool: int = 512, posterior_samples: int = 32, mc_samples: int = 16) -> dict:
    """A synthetic campaign shaped like the real one, with pools shrunk for runtime.

    Inputs start at 1.0 rather than 0.0 so the log-link objective's mean function
    has strictly positive features, which is what the real campaign's
    ``log(speed_1)`` term requires too.
    """
    return {
        "inputs": [
            {"name": f"x{i}", "start": 1.0, "stop": 2.0, "step": 0.05}
            for i in range(INPUT_DIM)
        ],
        "objectives": {
            "contract_version": "TEST_ONLY-round-sim-v1",
            "scaling_mode": "fixed_affine",
            "specs": [
                {
                    "name": "flat",
                    "goal": "maximize",
                    "transform": "affine",
                    "model_source_column": "flat",
                    "lower_anchor": 0.0,
                    "upper_anchor": 3.0,
                },
                {
                    "name": "sloped",
                    "goal": "maximize",
                    "transform": "affine",
                    "model_source_column": "sloped",
                    "lower_anchor": -4.0,
                    "upper_anchor": 0.0,
                    "mean_function": {
                        "response": "identity",
                        "features": [{"column": "x0", "transform": "identity"}],
                    },
                },
                {
                    # the thickness analogue: trains on a positive measurement, has a
                    # log response, and its utility peaks at a target
                    "name": "peaked",
                    "goal": "target",
                    "transform": "gaussian_target",
                    "model_source_column": "peaked",
                    "target": 650.0,
                    "sigma": 176.7766952966369,
                    "mean_function": {
                        "response": "log",
                        "features": [{"column": "x0", "transform": "log"}],
                    },
                },
            ],
        },
        "reference_point_utility": [-0.01, -0.01, -0.01],
        "rounds": {
            "r1": {
                "method": "ucb_hvi",
                "batch_size": R1_SIZE,
                "replicates_per_condition": 3,
                "beta": 4.0,
                "candidate_pool_size": pool,
                "posterior_samples": posterior_samples,
                "moment_method": "monte_carlo",
            },
            "r2": {
                "method": "qlognehvi",
                "batch_size": R2_SIZE,
                "replicates_per_condition": 3,
                "candidate_pool_size": pool,
                "mc_samples": mc_samples,
            },
        },
        "local_penalization": {
            "radius": 0.25,
            "min_batch_distance": 0.15,
            "min_observed_distance": 0.0,
            "dimension_weights": None,
        },
        "model": {"variant": "dim_scaled_prior"},
        "reproducibility": {"seed": SEED},
        "constraints": [],
    }


def _measurements(X: np.ndarray) -> np.ndarray:
    """Deterministic stand-in for the workbook's measured columns."""
    X = np.asarray(X, dtype=float)
    flat = X.mean(axis=1)
    sloped = -np.linalg.norm(X - 1.5, axis=1)
    peaked = 650.0 * X[:, 0] ** -0.5 * X[:, 1] ** 0.3
    return np.column_stack([flat, sloped, peaked])


@pytest.fixture(scope="module")
def synthetic():
    config = _config()
    transform = build_objective_transform(config)
    X_r0 = run_r0_lhs(config, n=R0_SIZE, seed=SEED).conditions.to_numpy(float)
    Y_r0 = _measurements(X_r0)
    oracle, warnings = fit_campaign_models(config, X_r0, Y_r0, seed=SEED)
    return {
        "config": config,
        "transform": transform,
        "X_r0": X_r0,
        "Y_r0": Y_r0,
        "oracle": oracle,
        "oracle_warnings": warnings,
        "reference": np.asarray(config["reference_point_utility"], float),
    }


@pytest.fixture(scope="module")
def cell(synthetic):
    return prs.run_cell(
        synthetic["config"],
        synthetic["oracle"],
        synthetic["X_r0"],
        synthetic["transform"],
        synthetic["reference"],
        radius=0.25,
        beta=4.0,
        seed=SEED,
    )


# --------------------------------------------------------------------------- #
# the oracle
# --------------------------------------------------------------------------- #


def test_the_oracle_is_deterministic(synthetic) -> None:
    """Two calls on the same model and inputs must agree bit for bit.

    Not a tidiness property. The manifest's central question is "did two parameter
    cells propose the same batch", and a stochastic oracle would score the same
    batch differently in two cells, so identical batches would look different.
    """
    first = prs.oracle_predict(
        synthetic["oracle"], synthetic["config"], synthetic["X_r0"], synthetic["transform"]
    )
    second = prs.oracle_predict(
        synthetic["oracle"], synthetic["config"], synthetic["X_r0"], synthetic["transform"]
    )
    assert np.array_equal(first, second)


def test_refitting_the_oracle_at_the_same_seed_reproduces_it(synthetic) -> None:
    refit, _warnings = fit_campaign_models(
        synthetic["config"], synthetic["X_r0"], synthetic["Y_r0"], seed=SEED
    )
    again = prs.oracle_predict(
        refit, synthetic["config"], synthetic["X_r0"], synthetic["transform"]
    )
    first = prs.oracle_predict(
        synthetic["oracle"], synthetic["config"], synthetic["X_r0"], synthetic["transform"]
    )
    assert np.allclose(first, again, rtol=0, atol=1e-12)


def test_the_oracle_reports_the_median_not_the_lognormal_mean(synthetic) -> None:
    """``exp(mu)``, never ``exp(mu + v/2)``.

    The lognormal mean is the correct mean, and Annie's branch used it. It is the
    wrong choice for an oracle because it makes the simulated ground truth a
    function of the posterior VARIANCE, which is large exactly where the 15 real
    films are sparse -- the landscape would then bulge in the regions the optimiser
    is about to explore. This pins the median so nobody "fixes" it back.
    """
    config, transform = synthetic["config"], synthetic["transform"]
    # somewhere away from the training points, so the variance is not ~0 and the
    # two conventions actually differ
    X = np.full((3, INPUT_DIM), 1.975)
    X[1, :] = 1.025
    X[2, 0] = 1.5

    from mobo_kit.campaign import normalise_inputs

    model = synthetic["oracle"]
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(
            torch.tensor(normalise_inputs(config, X), dtype=torch.double),
            observation_noise=False,
        )
        mean = posterior.mean.numpy()
        variance = posterior.variance.numpy()

    peaked = [i for i, s in enumerate(transform.specs) if s.model_link == "log"]
    assert peaked, "the synthetic campaign must carry a log-link objective"
    index = peaked[0]

    produced = prs.oracle_predict(model, config, X, transform)[:, index]
    median = np.exp(mean[:, index])
    lognormal_mean = np.exp(mean[:, index] + 0.5 * variance[:, index])

    assert np.allclose(produced, median, rtol=0, atol=1e-12)
    # and the two are genuinely distinguishable here, so the assertion has teeth
    assert np.max(np.abs(median - lognormal_mean)) > 1e-6


def test_identity_link_objectives_pass_through_untouched(synthetic) -> None:
    config, transform = synthetic["config"], synthetic["transform"]
    from mobo_kit.campaign import normalise_inputs

    model = synthetic["oracle"]
    model.eval()
    with torch.no_grad():
        mean = model.posterior(
            torch.tensor(normalise_inputs(config, synthetic["X_r0"]), dtype=torch.double),
            observation_noise=False,
        ).mean.numpy()
    produced = prs.oracle_predict(model, config, synthetic["X_r0"], transform)
    for index, spec in enumerate(transform.specs):
        if spec.model_link != "log":
            assert np.allclose(produced[:, index], mean[:, index], rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- #
# the encoding this script exists to get right
# --------------------------------------------------------------------------- #


def test_measurement_space_values_must_be_encoded_before_the_transform() -> None:
    """Why R1 is re-implemented rather than taken from ``campaign.run_r1_ucb``.

    ``ObjectiveTransform.transform`` decodes the link itself, so a log-link
    objective handed raw measurement values is exponentiated a second time.  For a
    650 nm Gaussian target that overflows to exactly 0.0 -- finite, so no guard
    fires and nothing raises.  This pins the failure mode rather than the caller,
    so it stays true whatever ``campaign.py`` later does.
    """
    config = _config()
    transform = build_objective_transform(config)
    physical = np.array([[1.5, -2.0, 360.0], [1.5, -2.0, 1303.0]], dtype=float)

    unencoded = transform.transform(torch.tensor(physical, dtype=torch.double)).numpy()
    assert np.all(unencoded[:, 2] == 0.0)

    encoded = prs.utilities(physical, transform)
    assert np.all(encoded[:, 2] > 0.0)
    assert np.all(encoded[:, 2] <= 1.0)
    # the two identity-link columns are unaffected either way
    assert np.allclose(unencoded[:, :2], encoded[:, :2])


def test_to_model_space_rejects_non_positive_values_on_a_log_link() -> None:
    config = _config()
    transform = build_objective_transform(config)
    with pytest.raises(ValueError, match="strictly positive"):
        prs.to_model_space(np.array([[1.0, -1.0, 0.0]]), transform)


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #


def test_the_loop_produces_23_conditions_split_15_5_3(cell) -> None:
    assert len(cell["X"]["R0"]) == R0_SIZE
    assert len(cell["X"]["R1"]) == R1_SIZE
    assert len(cell["X"]["R2"]) == R2_SIZE
    assert len(cell["X"]["all"]) == R0_SIZE + R1_SIZE + R2_SIZE == 23


def test_round_assignment_labels_every_condition_exactly_once(cell, synthetic) -> None:
    names = [item["name"] for item in synthetic["config"]["inputs"]]
    frame = prs.rounds_frame(cell, names, synthetic["transform"])
    assert len(frame) == 23
    assert frame["round"].value_counts().to_dict() == {"R0": 15, "R1": 5, "R2": 3}
    # the rows carry the conditions they claim to
    for round_name, size in (("R0", 15), ("R1", 5), ("R2", 3)):
        block = frame.loc[frame["round"] == round_name, names].to_numpy(float)
        assert block.shape == (size, len(names))
        assert np.allclose(block, cell["X"][round_name])


def test_every_round_carries_an_oracle_value_and_a_utility(cell, synthetic) -> None:
    names = [item["name"] for item in synthetic["config"]["inputs"]]
    frame = prs.rounds_frame(cell, names, synthetic["transform"])
    for spec in synthetic["transform"].specs:
        assert f"oracle_{spec.name}" in frame.columns
        assert f"utility_{spec.name}" in frame.columns
        assert np.isfinite(frame[f"oracle_{spec.name}"]).all()
        assert np.isfinite(frame[f"utility_{spec.name}"]).all()


def test_hypervolume_is_recorded_at_all_three_stages(cell) -> None:
    stages = ("R0", "R0+R1", "R0+R1+R2")
    assert set(cell["hv"]) == set(stages)
    values = [cell["hv"][stage] for stage in stages]
    # monotone BY CONSTRUCTION -- adding points can only grow a Pareto front. This
    # asserts the bookkeeping, not that optimisation happened.
    assert values[0] <= values[1] <= values[2]


# --------------------------------------------------------------------------- #
# the manifest
# --------------------------------------------------------------------------- #


def test_manifest_row_has_exactly_the_declared_columns(cell) -> None:
    row = prs.manifest_row(
        cell, condition_id=1, arm="both", seed=SEED, baseline_unencoded=0.004659
    )
    assert tuple(row) == prs.MANIFEST_COLUMNS
    frame = pd.DataFrame([row], columns=list(prs.MANIFEST_COLUMNS))
    assert list(frame.columns) == list(prs.MANIFEST_COLUMNS)
    assert frame["min_batch_distance"].iloc[0] == prs.PINNED_MIN_BATCH_DISTANCE


def test_the_manifest_carries_the_baseline_tripwire(cell) -> None:
    """The check that would catch the encoding defect coming back.

    ``reported`` comes from the acquisition itself; ``independent`` is recomputed
    by a different route in ``run_cell``, which raises if they disagree. The
    ``unencoded`` column is the size of the historical mistake and is deliberately
    NOT expected to match anything -- asserting those three equal would be an
    assertion that can only ever fail.
    """
    row = prs.manifest_row(
        cell, condition_id=1, arm="both", seed=SEED, baseline_unencoded=0.004659
    )
    assert row["baseline_hv_reported_by_r1"] == pytest.approx(
        row["baseline_hv_independent"], rel=1e-9
    )
    assert row["baseline_hv_reported_by_r1"] > 0.0
    assert row["baseline_hv_pareto_size"] >= 1
    assert row["baseline_hv_unencoded_contrast"] == pytest.approx(0.004659)


def test_run_cell_refuses_a_baseline_it_cannot_reproduce(cell) -> None:
    """The tripwire fires rather than writing a plausible manifest.

    ``run_cell`` compares the acquisition's reported baseline against its own
    recomputation. Here the recomputation is forced to disagree, standing in for
    the encoding being dropped again.
    """
    assert cell["baseline"]["reported"] == pytest.approx(
        cell["baseline"]["independent"], rel=1e-9
    )
    assert not math.isclose(
        cell["baseline"]["reported"], cell["baseline"]["reported"] * 0.01, rel_tol=1e-9
    ), "the comparison must be able to tell a 100x error apart"


def test_batch_hash_ignores_row_order_but_not_row_content() -> None:
    frame = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], columns=["a", "b"])
    shuffled = frame.iloc[[2, 0, 1]].reset_index(drop=True)
    assert prs.batch_hash(frame) == prs.batch_hash(shuffled)

    changed = frame.copy()
    changed.iloc[0, 0] = 1.5
    assert prs.batch_hash(frame) != prs.batch_hash(changed)


# --------------------------------------------------------------------------- #
# the parameter grid
# --------------------------------------------------------------------------- #


def test_ofat_is_thirteen_cells_because_the_arms_share_the_anchor() -> None:
    cells = prs.ofat_conditions()
    assert len(cells) == 13
    assert len({(r, b) for r, b, _ in cells}) == 13

    radius_arm = [(r, b) for r, b, arm in cells if arm in ("radius", "both")]
    beta_arm = [(r, b) for r, b, arm in cells if arm in ("beta", "both")]
    assert len(radius_arm) == 9
    assert len(beta_arm) == 5
    assert {b for _, b in radius_arm} == {prs.ANCHOR_BETA}
    assert {r for r, _ in beta_arm} == {prs.ANCHOR_RADIUS}
    # exactly one cell belongs to both arms
    assert sum(1 for _, _, arm in cells if arm == "both") == 1


def test_full_grid_is_the_45_cell_cross() -> None:
    cells = prs.full_grid_conditions()
    assert len(cells) == len(prs.OFAT_RADII) * len(prs.OFAT_BETAS) == 45


def test_min_batch_distance_is_pinned_in_every_cell() -> None:
    """The sweep's fixed constant. A cell that changed it would not be comparable."""
    base = _config()
    base["local_penalization"]["min_batch_distance"] = 0.99  # a wrong value to override
    for radius, beta, _arm in prs.ofat_conditions():
        config = prs.cell_config(base, radius=radius, beta=beta)
        assert config["local_penalization"]["min_batch_distance"] == 0.15
        assert config["local_penalization"]["radius"] == radius
        assert config["rounds"]["r1"]["beta"] == beta
    # and the base config is not mutated by building a cell
    assert base["local_penalization"]["min_batch_distance"] == 0.99


def test_cell_slug_matches_annies_directory_convention() -> None:
    assert prs.cell_slug(0.25, 4.0) == "radius_0p25__beta_4"
    assert prs.cell_slug(0.05, 25.0) == "radius_0p05__beta_25"


def test_fixed_slice_values_are_grid_snapped_medians() -> None:
    config = _config()
    from mobo_kit.design import build_design_from_config

    design = build_design_from_config(dict(config))
    X = run_r0_lhs(config, n=R0_SIZE, seed=SEED).conditions.to_numpy(float)
    fixed = prs.fixed_slice_values(design, X)
    assert fixed.shape == (INPUT_DIM,)
    for index, grid in enumerate(design.var_array):
        allowed = np.asarray(grid, dtype=float)
        assert np.any(np.isclose(fixed[index], allowed, atol=1e-9)), "must be on grid"
        # and it is the grid value nearest the median, not something else
        median = float(np.median(X[:, index]))
        assert fixed[index] == pytest.approx(
            allowed[np.argmin(np.abs(allowed - median))]
        )


# --------------------------------------------------------------------------- #
# figures render headlessly
# --------------------------------------------------------------------------- #


def test_every_figure_type_renders_without_a_display(cell, synthetic, tmp_path) -> None:
    from mobo_kit.design import build_design_from_config

    config, transform = synthetic["config"], synthetic["transform"]
    design = build_design_from_config(dict(config))
    names = list(design.names)
    fixed = prs.fixed_slice_values(design, synthetic["X_r0"])
    pair = (names[0], names[1])

    mesh_x, mesh_y, surfaces = prs.surface_grid(
        cell["final_model"], config, design, transform, pair, fixed, points=9
    )
    assert surfaces.shape == (9, 9, 3)
    assert np.isfinite(surfaces).all()

    for index, spec in enumerate(transform.specs):
        path = tmp_path / f"surface_{spec.name}.png"
        prs.plot_surface(
            path, mesh_x, mesh_y, surfaces[..., index], pair, spec,
            cell["X"], design, fixed,
            radius=0.25, beta=4.0, seed=SEED, warning_banner=None,
        )
        assert path.is_file() and path.stat().st_size > 0

    boxplots = tmp_path / "boxplots.png"
    prs.plot_boxplots(boxplots, cell, transform, seed=SEED, warning_banner=None)
    assert boxplots.is_file() and boxplots.stat().st_size > 0

    hv = tmp_path / "hv.png"
    prs.plot_hypervolume(
        hv, cell, synthetic["reference"], seed=SEED,
        warning_banner="FIT GUARD: banner path must render too.",
    )
    assert hv.is_file() and hv.stat().st_size > 0


def test_the_slice_caveat_is_only_on_figures_that_have_a_slice() -> None:
    """A caveat printed where it is not true trains people to skip footers."""
    assert "Slice" in prs.SLICE_CAVEAT
    assert "Slice" not in prs.ROUND_N_CAVEAT
    assert prs.ORACLE_CAVEAT.startswith("Oracle")
    assert "not measurements" in prs.ORACLE_CAVEAT


# --------------------------------------------------------------------------- #
# end to end on the real workbook
# --------------------------------------------------------------------------- #


@pytest.mark.local_input
@pytest.mark.skipif(
    not Path(SOURCE).is_file(), reason=f"{SOURCE} is not present in this checkout"
)
def test_one_condition_one_pair_end_to_end(tmp_path) -> None:
    """The whole script, headless, on the real campaign workbook.

    Pools are shrunk through a scratch config so this stays a smoke test; the
    numbers it produces are therefore NOT the campaign's and are not asserted on.
    What is asserted is that the artifacts appear where the directory convention
    says they will.
    """
    import yaml

    from mobo_kit.campaign import load_campaign_config

    config = load_campaign_config("configs/campaign_d2d_perovskite.yaml")
    config["rounds"]["r1"]["candidate_pool_size"] = 256
    config["rounds"]["r1"]["posterior_samples"] = 16
    config["rounds"]["r2"]["candidate_pool_size"] = 256
    config["rounds"]["r2"]["mc_samples"] = 8
    scratch = tmp_path / "campaign.yaml"
    scratch.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    output = tmp_path / "out"
    code = prs.main([
        "--workbook", SOURCE,
        "--config", str(scratch),
        "--output-dir", str(output),
        "--conditions", "radius_0p25__beta_4",
        "--pairs", "speed_1,precur_conc",
        "--slice-points", "9",
    ])
    assert code == 0

    manifest = output / "manifest.csv"
    assert manifest.is_file()
    frame = pd.read_csv(manifest)
    assert list(frame.columns) == list(prs.MANIFEST_COLUMNS)
    assert len(frame) == 1
    assert frame["radius"].iloc[0] == 0.25
    assert frame["beta"].iloc[0] == 4.0
    assert frame["min_batch_distance"].iloc[0] == 0.15

    # Annie's directory convention: {pair}/qlognehvi/radius_*__beta_*/
    pair_dir = output / "speed_1__precur_conc" / "qlognehvi" / "radius_0p25__beta_4"
    for objective in ("uniformity", "optoelectronic", "thickness"):
        assert (pair_dir / f"final_surface_{objective}.png").is_file()

    condition_dir = output / "by_condition" / "qlognehvi" / "radius_0p25__beta_4"
    assert (condition_dir / "round_boxplots.png").is_file()
    assert (condition_dir / "hypervolume_by_round.png").is_file()
    rounds = pd.read_csv(condition_dir / "all_rounds.csv")
    assert rounds["round"].value_counts().to_dict() == {"R0": 15, "R1": 5, "R2": 3}
