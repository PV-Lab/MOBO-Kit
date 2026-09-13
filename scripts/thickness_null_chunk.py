"""One shard of the thickness permutation null.

Emits JSON so shards can be pooled into a single null distribution:

    python scripts/thickness_null_chunk.py --n 250 --seed 0 --out shard0.json

Each shard permutes the nm measurements, redoes the full leave-one-out fit
(refitting the structured linear mean inside every fold, so the null is not
flattered), applies the utility transform, and records the resulting Spearman
and R2.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from thickness_permutation_and_mean import (  # noqa: E402
    expected_score,
    load,
    loo_nm,
    r2,
)

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workbook", default="local_inputs/Summary Table.xlsx")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--structured", action="store_true", default=True)
    args = ap.parse_args()

    X, Xphys, T_nm, score = load(Path(args.workbook))
    n = len(T_nm)
    rng = np.random.default_rng(args.seed)

    rhos, r2s = [], []
    for _ in range(args.n):
        perm = rng.permutation(n)
        mu_p, var_p = loo_nm(
            X, T_nm[perm], structured_mean=args.structured, Xphys=Xphys
        )
        e_p = expected_score(mu_p, var_p)
        rhos.append(float(spearmanr(score[perm], e_p).statistic))
        r2s.append(float(r2(score[perm], e_p)))

    Path(args.out).write_text(
        json.dumps({"seed": args.seed, "n": args.n, "rho": rhos, "r2": r2s}),
        encoding="utf-8",
    )
    print(
        f"shard seed={args.seed} n={args.n} mean_rho={np.mean(rhos):+.4f} -> {args.out}"
    )


if __name__ == "__main__":
    main()
