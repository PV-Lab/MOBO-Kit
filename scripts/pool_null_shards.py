"""Pool permutation-null shards into one p-value with a binomial interval.

    python scripts/pool_null_shards.py --observed-rho 0.4607 --observed-r2 -0.0191

The binomial interval matters: at 200 shuffles a p of 0.020 is about 4
exceedances, whose 95% interval reaches 0.05. More shuffles shrink that, and the
interval is what says whether the p-value is safe to quote.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta


def clopper_pearson(successes: int, trials: int, alpha: float = 0.05):
    lower = (
        0.0
        if successes == 0
        else beta.ppf(alpha / 2, successes, trials - successes + 1)
    )
    upper = (
        1.0
        if successes == trials
        else beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    )
    return float(lower), float(upper)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", default="local_outputs/null_shards/shard_*.json")
    ap.add_argument("--observed-rho", type=float, required=True)
    ap.add_argument("--observed-r2", type=float, default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.shards))
    if not paths:
        raise SystemExit(f"no shards matched {args.shards!r}")

    rho: list[float] = []
    r2: list[float] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rho.extend(payload["rho"])
        r2.extend(payload["r2"])
        print(
            f"  {Path(path).name}: n={len(payload['rho'])} "
            f"mean_rho={np.mean(payload['rho']):+.4f}"
        )

    rho_a = np.array(rho)
    n = len(rho_a)
    print(f"\npooled shuffles: {n} from {len(paths)} shard(s)")

    for label, null, observed in (
        ("Spearman", rho_a, args.observed_rho),
        ("R2", np.array(r2), args.observed_r2),
    ):
        if observed is None:
            continue
        exceed = int((null >= observed).sum())
        p = exceed / len(null)
        lo, hi = clopper_pearson(exceed, len(null))
        verdict = (
            "CLEARS p<0.05"
            if hi < 0.05
            else (
                "significant but interval touches 0.05"
                if p < 0.05
                else "not significant"
            )
        )
        print(f"\n  {label}")
        print(f"    observed        {observed:+.4f}")
        print(f"    null mean/sd    {null.mean():+.4f} / {null.std():.4f}")
        print(f"    95th percentile {np.percentile(null, 95):+.4f}")
        print(f"    exceedances     {exceed}/{len(null)}")
        print(f"    p               {p:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   {verdict}")


if __name__ == "__main__":
    main()
