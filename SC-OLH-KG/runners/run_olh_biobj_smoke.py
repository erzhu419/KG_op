"""Run the bi-objective OLH-KG smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.biobj_smoke import BiObjectiveOLHKGSmoke, BiObjSmokeConfig  # noqa: E402
from problems.rzdt import make_problem  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="RegimeRZDT1")
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--n0", type=int, default=6)
    parser.add_argument("--K1", type=int, default=20)
    parser.add_argument("--variance_mode", default="class",
                        choices=["pooled", "oracle", "class", "orthogonal", "factor"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    problem = make_problem(args.problem)
    config = BiObjSmokeConfig(
        N=args.N,
        n0=args.n0,
        K1=args.K1,
        variance_mode=args.variance_mode,
        seed=args.seed,
    )
    alg = BiObjectiveOLHKGSmoke(problem, config)
    result = alg.run(verbose=args.verbose)
    print(json.dumps(result, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
