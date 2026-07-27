from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mobo_kit.candidate_pool import sample_discrete_candidate_pool
from mobo_kit.d2d_campaign import (
    D2D_INPUT_COLUMNS,
    D2D_OBJECTIVE_COLUMNS,
    D2D_WORKBOOK_INPUT_COLUMNS,
    aggregate_replicate_objectives,
    combine_r0_and_aggregated_r1,
    load_d2d_debug_config,
    prepare_d2d_training_data,
)
from mobo_kit.d2d_r2_test import (
    D2D_R2_TEST_WATERMARK,
    propose_d2d_qlognehvi_test_batch,
)
from mobo_kit.data import x_normalizer_np
from mobo_kit.models import fit_gp_models


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "d2d_step2b_debug.yaml"


def _grid_rows(design, start: int, count: int) -> np.ndarray:
    rows = []
    for index in range(start, start + count):
        rows.append(
            [
                float(grid[(index * (dimension + 1) + 2 * dimension) % len(grid)])
                for dimension, grid in enumerate(design.var_array)
            ]
        )
    result = np.asarray(rows, dtype=float)
    assert np.unique(result, axis=0).shape[0] == count
    return result


def _synthetic_scores(X_norm: np.ndarray) -> np.ndarray:
    uniformity = 0.10 + 0.75 * (0.65 * X_norm[:, 0] + 0.35 * (1.0 - X_norm[:, 1]))
    optoelectronic = -9.4 + 1.8 * (0.55 * X_norm[:, 2] + 0.45 * X_norm[:, 4])
    thickness = 0.05 + 0.90 * np.exp(-(((X_norm[:, 5] - 0.55) / 0.32) ** 2))
    return np.column_stack([uniformity, optoelectronic, thickness])


def test_future_r2_uses_20_conditions_and_returns_three_synthetic_points() -> None:
    config = load_d2d_debug_config(PUBLIC_CONFIG_PATH, allow_public_template=True)
    X_r0 = _grid_rows(config.design, 0, 15)
    exception = config.off_grid_exceptions[0]
    X_r0[0, D2D_INPUT_COLUMNS.index(exception.input_name)] = exception.observed_value
    Y_r0 = _synthetic_scores(x_normalizer_np(X_r0, config.design))
    frame = pd.DataFrame({"Sample number": config.expected_sample_ids})
    for index, column in enumerate(D2D_WORKBOOK_INPUT_COLUMNS):
        frame[column] = X_r0[:, index]
    for index, column in enumerate(D2D_OBJECTIVE_COLUMNS):
        frame[column] = Y_r0[:, index]
    r0_training = prepare_d2d_training_data(frame, config)

    X_r1 = _grid_rows(config.design, 40, 5)
    Y_r1 = _synthetic_scores(x_normalizer_np(X_r1, config.design))
    records = []
    for candidate_index, (inputs, objectives) in enumerate(zip(X_r1, Y_r1), start=1):
        for replicate, offset in enumerate((-0.01, 0.0, 0.01), start=1):
            row = {
                "execution_id": f"R1-C{candidate_index:02d}-R{replicate}",
                "replicate_group": f"R1-C{candidate_index:02d}",
                "replicate_number": replicate,
            }
            row.update(dict(zip(D2D_INPUT_COLUMNS, inputs)))
            row.update(
                {
                    objective: float(value + offset * (objective_index + 1) / 10)
                    for objective_index, (objective, value) in enumerate(
                        zip(D2D_OBJECTIVE_COLUMNS, objectives)
                    )
                }
            )
            records.append(row)
    aggregated = aggregate_replicate_objectives(pd.DataFrame(records))
    assert aggregated.frame.shape[0] == 5
    assert aggregated.frame["complete_replicate_set"].all()

    X_phys, Y = combine_r0_and_aggregated_r1(r0_training, aggregated)
    assert X_phys.shape == (20, 10)
    assert Y.shape == (20, 3)
    X_norm = x_normalizer_np(X_phys, config.design)
    train_X = torch.as_tensor(X_norm, dtype=torch.double)
    train_Y = torch.as_tensor(Y, dtype=torch.double)
    torch.manual_seed(config.seed)
    model = fit_gp_models(train_X, train_Y)

    pool = sample_discrete_candidate_pool(
        config.design,
        64,
        seed=911,
        observed_phys=X_phys[1:],
        row_constraints=[],
    )
    with pytest.raises(ValueError, match="synthetic test-only"):
        propose_d2d_qlognehvi_test_batch(
            pool, model, train_X, config, test_only=False, mc_samples=16
        )
    for invalid_samples in (True, 3.5, 0, -1):
        with pytest.raises(ValueError, match="mc_samples must be a positive"):
            propose_d2d_qlognehvi_test_batch(
                pool,
                model,
                train_X,
                config,
                test_only=True,
                mc_samples=invalid_samples,
            )
    for invalid_seed in (True, 3.5, -1):
        with pytest.raises(ValueError, match="seed must be a non-negative"):
            propose_d2d_qlognehvi_test_batch(
                pool,
                model,
                train_X,
                config,
                test_only=True,
                mc_samples=16,
                seed=invalid_seed,
            )
    with pytest.raises(ValueError, match="off-grid"):
        propose_d2d_qlognehvi_test_batch(
            replace(pool, X_phys=pool.X_phys + 1e-6),
            model,
            train_X,
            config,
            test_only=True,
            mc_samples=16,
        )
    inconsistent_indices = pool.grid_indices.copy()
    inconsistent_indices[0, 0] = (inconsistent_indices[0, 0] + 1) % len(
        config.design.var_array[0]
    )
    with pytest.raises(ValueError, match="grid_indices do not match"):
        propose_d2d_qlognehvi_test_batch(
            replace(pool, grid_indices=inconsistent_indices),
            model,
            train_X,
            config,
            test_only=True,
            mc_samples=16,
        )
    inconsistent_norm = pool.X_norm.copy()
    inconsistent_norm[0, 0] += 1e-6
    with pytest.raises(ValueError, match="normalized rows do not match"):
        propose_d2d_qlognehvi_test_batch(
            replace(pool, X_norm=inconsistent_norm),
            model,
            train_X,
            config,
            test_only=True,
            mc_samples=16,
        )
    result = propose_d2d_qlognehvi_test_batch(
        pool,
        model,
        train_X,
        config,
        test_only=True,
        mc_samples=16,
        seed=73,
        chunk_size=64,
    )
    assert result.selection.X_phys.shape == (3, 10)
    assert np.unique(result.selection.X_phys, axis=0).shape[0] == 3
    assert result.metadata["objective_contract_version"] == (
        "d2d-step2b-debug-objectives-v1"
    )
    assert result.metadata["test_only"] is True
    assert result.metadata["synthetic_only"] is True
    assert result.metadata["approved_for_experiment"] is False
    assert result.metadata["candidate_status"] == D2D_R2_TEST_WATERMARK
    assert result.selection.method_diagnostics["test_only"] is True
