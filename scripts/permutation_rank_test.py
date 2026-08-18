"""Does an objective's declared mean function beat chance on RANK?

    python scripts/permutation_rank_test.py --objective thickness --permutations 1800

`scripts/intake_new_data.py` answers "does the mean function beat the null by more
than the resolution floor". When the answer is INCONCLUSIVE -- the swing is real
but smaller than what LOO R2 can resolve at this N -- that is not a verdict, it is
a statement that R2 cannot decide. This is the instrument that decides.

**Rank, not R2, because rank is what drives candidate selection.** The acquisition
ranks candidates; it never consumes R2. The first campaign settled its thickness
mean function on exactly this basis (p = 0.0350, 95% CI [0.0270, 0.0446] at 1800
shuffles) and recorded that the R2 swing was consistent with it and no more.

**The linear coefficients are refit inside every null fold too**, on the training
rows only, so the null is not flattered by a trend fitted to all the data.

Config-driven, so it works on any contract. It deliberately does NOT replace
`scripts/thickness_permutation_and_mean.py`, which hard-codes the first campaign's
column positions and grid and is kept as that campaign's reproducible record.

Sharding, because the cost is (permutations x N) GP fits:

    python scripts/permutation_rank_test.py --shards 12 --shard 0 --out <dir>   # x12
    python scripts/permutation_rank_test.py --combine <dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy.stats import spearmanr

from mobo_kit.campaign import (
    build_objective_transform,
    load_campaign_config,
    normalise_inputs,
    objective_names,
)
from mobo_kit.loocv import loo_predictions
from mobo_kit.structured_mean import mean_spec_from_config
from mobo_kit.workbook_io import read_campaign_workbook


def measured_utility(transform: Any, index: int, Y_measured: np.ndarray) -> np.ndarray:
    """Utility of the measurements themselves, one column.

    `transform_measurements`, NOT `expected_transform`: the transform decodes the
    link itself, so handing it measurement-space nanometres exponentiates a value
    that was never a logarithm. That mistake cost this project an R1 batch, and it
    reappeared in the first draft of this script -- caught only because saturating
    the 650 nm Gaussian to 0.0 made the column constant and Spearman undefined. It
    is the third route by which the same defect has arrived; use the safe call.
    """
    block = torch.tensor(np.asarray(Y_measured, dtype=float), dtype=torch.double)
    return transform.transform_measurements(block)[:, index].detach().cpu().numpy()


def expected_utility(
    transform: Any,
    index: int,
    mu: np.ndarray,
    var: np.ndarray,
    Y_model_context: np.ndarray,
) -> np.ndarray:
    """E[utility] for one objective, through the campaign's own transform.

    `mu`/`var` are MODEL-space posterior moments for objective `index`. The other
    columns are filled with the measured model-space values at zero variance: the
    transform is elementwise per objective, so they cannot affect the column read
    back, and using real values rather than zeros keeps every column inside its
    own link's domain.
    """
    mean_block = torch.tensor(
        np.asarray(Y_model_context, dtype=float), dtype=torch.double
    ).clone()
    var_block = torch.zeros_like(mean_block)
    mean_block[:, index] = torch.tensor(mu, dtype=torch.double)
    var_block[:, index] = torch.tensor(var, dtype=torch.double)
    utility = transform.expected_transform(mean_block, var_block)
    return utility[:, index].detach().cpu().numpy()


def _setup(args: argparse.Namespace) -> dict[str, Any]:
    config = load_campaign_config(args.config)
    names = list(objective_names(config))
    if args.objective not in names:
        raise SystemExit(f"--objective must be one of {names}; got {args.objective!r}")
    index = names.index(args.objective)
    entry = config["objectives"]["specs"][index]
    mean_spec = mean_spec_from_config(entry)
    if mean_spec is None:
        raise SystemExit(
            f"{args.objective!r} declares no mean_function, so there is nothing to "
            "adjudicate."
        )

    contents = read_campaign_workbook(args.workbook, config)
    if contents.errors:
        raise SystemExit("The workbook has errors; refusing to test a subset.")

    transform = build_objective_transform(config)
    design_names = [item["name"] for item in config["inputs"]]
    return {
        "config": config,
        "index": index,
        "entry": entry,
        "mean_spec": mean_spec,
        "transform": transform,
        "design_names": design_names,
        "lowers": np.array([float(i["start"]) for i in config["inputs"]]),
        "uppers": np.array([float(i["stop"]) for i in config["inputs"]]),
        "X_phys": contents.inputs.to_numpy(float),
        "X_norm": normalise_inputs(config, contents.inputs.to_numpy(float)),
        "y": contents.model_values[args.objective].to_numpy(float),
        "Y_measured": contents.model_values.to_numpy(float),
    }


def _rho(state: Mapping[str, Any], y: np.ndarray, *, seed: int) -> float:
    """LOO expected-utility rank correlation against the measured utility.

    Both sides are in UTILITY space, which is what the acquisition ranks. Under a
    permutation the substituted column travels through the same transform as the
    real one, so the null is built on the same quantity as the observation.
    """
    transform, index = state["transform"], state["index"]
    Y_measured = state["Y_measured"].copy()
    Y_measured[:, index] = y

    # the shared fold loop, so this null is built on exactly the numbers
    # intake reports and the round report plots
    result = loo_predictions(state["config"], state["entry"], state["X_phys"], y, seed=seed)
    mu, var = result.mean_model_space, result.variance_model_space
    context = (
        transform.encode_measurements(
            torch.tensor(Y_measured, dtype=torch.double)
        )
        .detach()
        .cpu()
        .numpy()
    )
    predicted = expected_utility(transform, index, mu, var, context)
    measured = measured_utility(transform, index, Y_measured)
    return float(spearmanr(measured, predicted).statistic)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/campaign_d2d_perovskite_test.yaml")
    parser.add_argument("--workbook", default="local_inputs/Summary Table Test.xlsx")
    parser.add_argument("--objective", default="thickness")
    parser.add_argument("--permutations", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fit-seed", type=int, default=73)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--out", default=None, help="directory for shard results")
    parser.add_argument("--combine", default=None, help="combine shards in this dir")
    args = parser.parse_args(argv)

    if args.combine:
        return combine(Path(args.combine))

    state = _setup(args)
    y = state["y"]
    n = len(y)

    observed = _rho(state, y, seed=args.fit_seed)
    print(f"objective          {args.objective}")
    print(f"rows               {n}")
    print(f"observed rank rho  {observed:+.4f}")

    # Every shard draws the SAME permutation stream and takes a slice of it, so the
    # union of shards is exactly the single-process run and shard boundaries cannot
    # change the answer.
    rng = np.random.default_rng(args.seed)
    permutations = [rng.permutation(n) for _ in range(args.permutations)]
    mine = [
        (i, order)
        for i, order in enumerate(permutations)
        if i % args.shards == args.shard
    ]
    print(f"shard {args.shard}/{args.shards}   {len(mine)} of {args.permutations} shuffles")

    null_rhos: list[float] = []
    for position, (i, order) in enumerate(mine, start=1):
        null_rhos.append(_rho(state, y[order], seed=args.fit_seed))
        if position % 10 == 0 or position == len(mine):
            exceed = sum(1 for r in null_rhos if r >= observed)
            print(f"  {position}/{len(mine)}  exceedances so far {exceed}", flush=True)

    payload = {
        "objective": args.objective,
        "n": n,
        "observed_rho": observed,
        "shard": args.shard,
        "shards": args.shards,
        "permutations_total": args.permutations,
        "null_rhos": null_rhos,
        "seed": args.seed,
        "fit_seed": args.fit_seed,
    }
    if args.out:
        directory = Path(args.out)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"shard_{args.shard:03d}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        print(f"wrote {path}")
    else:
        report(observed, null_rhos, args.permutations)
    return 0


def report(observed: float, null_rhos: list[float], total: int) -> None:
    """The p-value, with the interval that says how well it is resolved."""
    drawn = len(null_rhos)
    if drawn == 0:
        # --permutations 0 is a legitimate "just tell me the observed statistic"
        # mode; reporting a p-value from no shuffles would be inventing one.
        print("")
        print("no shuffles drawn, so there is no null and no p-value.")
        return
    exceed = sum(1 for r in null_rhos if r >= observed)
    # (exceed + 1) / (drawn + 1): the observed statistic is itself one draw from the
    # null under the null hypothesis, so a p-value of exactly 0 is not available and
    # claiming one would overstate the evidence.
    p = (exceed + 1) / (drawn + 1)
    se = math.sqrt(max(p * (1 - p) / drawn, 0.0))
    low, high = max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)
    print()
    print(f"shuffles drawn     {drawn} of {total}")
    print(f"exceedances        {exceed}")
    print(f"null mean rho      {np.mean(null_rhos):+.4f}")
    print(f"null sd            {np.std(null_rhos, ddof=1):.4f}")
    print(f"p                  {p:.4f}   95% CI [{low:.4f}, {high:.4f}]")
    print()
    if high < 0.05:
        print("VERDICT  KEEP. The interval clears 0.05, so the rank result is not")
        print("         chance and the mean function has earned its place.")
    elif p < 0.05:
        print("VERDICT  BORDERLINE. The point estimate clears 0.05 but the interval")
        print("         does not. Draw more shuffles before deciding.")
    else:
        print("VERDICT  DELETE. The rank result is inside what shuffling produces,")
        print("         so nothing distinguishes this mean function from chance.")


def combine(directory: Path) -> int:
    shards = sorted(directory.glob("shard_*.json"))
    if not shards:
        raise SystemExit(f"no shard_*.json under {directory}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in shards]
    observed = {round(p["observed_rho"], 12) for p in payloads}
    if len(observed) != 1:
        raise SystemExit(
            f"shards disagree on the observed statistic: {sorted(observed)}. They "
            "were not run against the same data."
        )
    null_rhos = [r for payload in payloads for r in payload["null_rhos"]]
    print(f"combined {len(shards)} shards, objective {payloads[0]['objective']}")
    print(f"observed rank rho  {payloads[0]['observed_rho']:+.4f}")
    report(payloads[0]["observed_rho"], null_rhos, payloads[0]["permutations_total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
