"""Oracle-only diagnostics for LF-OS representation quality.

This script never feeds oracle information back into OLH-KG.  It answers a
debugging question: if we are allowed to inspect the synthetic ground truth
offline, is LF-OS close enough to the correct low-dimensional structure that it
could work with few evaluations?  If yes, failures are likely candidate
selection/certification issues.  If no, the low-frequency assumptions or
hyperparameters need work before expensive KG sweeps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_quality import (  # noqa: E402
    json_safe,
    parse_csv,
    parse_weights,
    write_csv,
)
from problems.rzdt import make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402
from representation.orthogonal_sparse import (  # noqa: E402
    LowFrequencyOrthogonalSparsePolicyEncoder,
)


def parse_ints(text):
    return parse_csv(text, int)


def parse_floats(text):
    return parse_csv(text, float)


def unique(rows):
    seen = set()
    out = []
    for row in rows:
        key = tuple(int(v) for v in row)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def ridge_fit_predict(X, y, ridge):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    X_aug = np.column_stack([np.ones(len(X)), X])
    scale = np.std(X_aug, axis=0)
    scale[0] = 1.0
    scale = np.maximum(scale, 1e-12)
    Z = X_aug / scale[None, :]
    eye = np.eye(Z.shape[1], dtype=float)
    eye[0, 0] = 0.0
    beta = np.linalg.solve(Z.T @ Z + float(ridge) * eye, Z.T @ y)
    pred = Z @ beta
    return pred


def r2_score(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 1e-14:
        return 1.0
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def make_candidate_pool(problem, encoder, args, rng):
    rows = []
    source = {}

    def add(candidates, name):
        for x in candidates or []:
            key = tuple(int(v) for v in x)
            if key not in source:
                rows.append(key)
                source[key] = name

    add(problem.initial_samples(n=args.initial_count, rng=rng), "initial")
    add(problem.structured_candidates(n=args.structured_count, rng=rng), "structured")
    try:
        axis = list(problem.all_axis_solutions())
    except Exception:
        axis = []
    if axis and args.axis_cap > 0:
        if len(axis) > args.axis_cap:
            idx = np.linspace(0, len(axis) - 1, int(args.axis_cap))
            axis = [axis[int(round(i))] for i in idx]
        add(axis, "axis")
    if args.state_candidate_count > 0:
        add(
            encoder.inverse_candidates(
                n_anchors=args.state_candidate_count,
                inverse_pool_size=args.inverse_pool_size,
                inverse_neighbors=args.state_inverse_neighbors,
                rng=rng,
            ),
            "lf_os_inverse",
        )
    add([problem.sample_random(rng) for _ in range(args.random_count)], "random")
    true_best_x, true_best_obj = problem.true_best_feasible()
    return unique(rows), source, true_best_x, true_best_obj


def evaluate_pool(problem, rows):
    z_alpha = norm.ppf(1 - problem.alpha)
    obj = []
    margin = []
    feasible = []
    sigma = []
    for x in rows:
        obj_i = float(problem.true_objective(x))
        con_i = float(problem.true_constraint_mean(x))
        sig_i = float(problem.true_sigma(x)[1])
        mar_i = con_i + z_alpha * sig_i - float(problem.tau)
        obj.append(obj_i)
        margin.append(mar_i)
        feasible.append(mar_i <= 0.0)
        sigma.append(sig_i)
    return {
        "objective": np.asarray(obj, dtype=float),
        "chance_margin": np.asarray(margin, dtype=float),
        "feasible": np.asarray(feasible, dtype=bool),
        "sigma": np.asarray(sigma, dtype=float),
    }


def run_one(args, *, cutoff, active, floor, seed):
    rng = np.random.default_rng(int(seed))
    base = make_problem(
        args.problem,
        d=args.d,
        L=args.L,
        sigma=args.sigma,
        alpha=args.alpha,
    )
    problem = ScalarizedProblem(base, weights=parse_weights(args.weights))
    encoder = LowFrequencyOrthogonalSparsePolicyEncoder(
        problem,
        latent_dim=args.encoder_latent_dim,
        fit_pool_size=args.encoder_fit_pool_size,
        max_library_size=args.lf_os_max_library_size,
        low_frequency_components=cutoff,
        max_active=active,
        n_neighbors=args.lf_os_graph_neighbors,
        residual_floor_scale=floor,
        use_problem_state_anchor=not bool(args.disable_lf_os_problem_state_anchor),
        rng=rng,
    )
    rows, source, true_best_x, true_best_obj = make_candidate_pool(
        problem,
        encoder,
        args,
        rng,
    )
    eval_rows = list(rows)
    pool_contains_true_best = False
    true_best_pool_source = None
    if true_best_x is not None:
        true_best_x = tuple(int(v) for v in true_best_x)
        pool_contains_true_best = true_best_x in set(eval_rows)
        true_best_pool_source = source.get(true_best_x)
        if not pool_contains_true_best:
            eval_rows.append(true_best_x)
    y = evaluate_pool(problem, eval_rows)
    feats = encoder.features_many(eval_rows)
    non_oracle_mask = np.array([
        idx < len(rows) for idx, _ in enumerate(eval_rows)
    ], dtype=bool)
    train_idx = np.where(non_oracle_mask)[0]
    if len(train_idx) < max(4, feats.shape[1] + 1):
        train_idx = np.arange(len(eval_rows))
    pred_obj = ridge_fit_predict(feats[train_idx], y["objective"][train_idx], args.ridge)
    pred_margin = ridge_fit_predict(
        feats[train_idx],
        y["chance_margin"][train_idx],
        args.ridge,
    )
    full_pred_obj = ridge_fit_predict(feats, y["objective"], args.ridge)
    full_pred_margin = ridge_fit_predict(feats, y["chance_margin"], args.ridge)

    pool_feasible = y["feasible"][: len(rows)]
    pool_obj = y["objective"][: len(rows)]
    feasible_obj = pool_obj[pool_feasible]
    best_pool_obj = float(np.min(feasible_obj)) if len(feasible_obj) else np.inf
    pool_regret = (
        float(best_pool_obj - true_best_obj)
        if true_best_x is not None and np.isfinite(best_pool_obj)
        else None
    )
    if true_best_x is not None:
        true_feat = encoder.features(true_best_x)
        pool_feats = feats[: len(rows)]
        dist = np.linalg.norm(pool_feats - true_feat[None, :], axis=1)
        nearest = int(np.argmin(dist)) if len(dist) else -1
        nearest_dist = float(dist[nearest]) if nearest >= 0 else None
        nearest_source = source.get(rows[nearest], "unknown") if nearest >= 0 else None
        nearest_regret = (
            float(problem.true_objective(rows[nearest]) - true_best_obj)
            if nearest >= 0 else None
        )
    else:
        nearest_dist = None
        nearest_source = None
        nearest_regret = None

    diag = encoder.diagnostics()
    return {
        "problem": args.problem,
        "d": int(args.d),
        "seed": int(seed),
        "lf_os_low_frequency_components": int(cutoff),
        "lf_os_max_active": int(active),
        "lf_os_residual_floor_scale": float(floor),
        "n_pool": int(len(rows)),
        "n_eval": int(len(eval_rows)),
        "true_best_available": bool(true_best_x is not None),
        "pool_contains_true_best": bool(pool_contains_true_best),
        "true_best_pool_source": true_best_pool_source,
        "true_best_objective": None if true_best_x is None else float(true_best_obj),
        "pool_feasible_rate": float(np.mean(pool_feasible)) if len(pool_feasible) else None,
        "pool_best_feasible_regret": pool_regret,
        "nearest_true_best_lfos_distance": nearest_dist,
        "nearest_true_best_source": nearest_source,
        "nearest_true_best_regret": nearest_regret,
        "oracle_feature_r2_objective_train": r2_score(
            y["objective"][train_idx],
            pred_obj,
        ),
        "oracle_feature_r2_margin_train": r2_score(
            y["chance_margin"][train_idx],
            pred_margin,
        ),
        "oracle_feature_r2_objective_full": r2_score(y["objective"], full_pred_obj),
        "oracle_feature_r2_margin_full": r2_score(y["chance_margin"], full_pred_margin),
        "lf_os_active_dim": int(diag.get("active_dim", 0)),
        "lf_os_library_dim": int(diag.get("library_dim", 0)),
        "lf_os_mean_low_frequency_ratio": float(diag.get("mean_low_frequency_ratio", 0.0)),
        "lf_os_min_low_frequency_ratio": float(diag.get("min_low_frequency_ratio", 0.0)),
        "lf_os_residual_floor_mean": float(diag.get("residual_floor_mean", 0.0)),
        "lf_os_max_offdiag_gram": float(diag.get("max_offdiag_gram", 0.0)),
        "source_counts": {
            name: int(sum(1 for x in rows if source.get(x) == name))
            for name in sorted(set(source.values()))
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="HighDimStatePolicyRZDT1")
    parser.add_argument("--d", type=int, default=10000)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--encoder_latent_dim", type=int, default=8)
    parser.add_argument("--encoder_fit_pool_size", type=int, default=512)
    parser.add_argument("--lf_os_max_library_size", type=int, default=30)
    parser.add_argument("--lf_os_low_frequency_components_grid", default="5,8,12")
    parser.add_argument("--lf_os_max_active_grid", default="5,8")
    parser.add_argument("--lf_os_graph_neighbors", type=int, default=12)
    parser.add_argument("--lf_os_residual_floor_scale_grid", default="0.0,0.02,0.05")
    parser.add_argument("--disable_lf_os_problem_state_anchor", action="store_true")
    parser.add_argument("--initial_count", type=int, default=8)
    parser.add_argument("--structured_count", type=int, default=32)
    parser.add_argument("--axis_cap", type=int, default=0)
    parser.add_argument("--state_candidate_count", type=int, default=64)
    parser.add_argument("--state_inverse_neighbors", type=int, default=2)
    parser.add_argument("--inverse_pool_size", type=int, default=512)
    parser.add_argument("--random_count", type=int, default=1024)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--out_dir", default=str(ROOT / "profiles"))
    parser.add_argument("--out_prefix", default="")
    args = parser.parse_args()

    rows = []
    started = time.time()
    for seed in parse_ints(args.seeds):
        for cutoff in parse_ints(args.lf_os_low_frequency_components_grid):
            for active in parse_ints(args.lf_os_max_active_grid):
                for floor in parse_floats(args.lf_os_residual_floor_scale_grid):
                    row = run_one(
                        args,
                        cutoff=cutoff,
                        active=active,
                        floor=floor,
                        seed=seed,
                    )
                    rows.append(row)
                    print(
                        "[lfos-oracle-gap] "
                        f"seed={seed} lf={cutoff} active={active} floor={floor} "
                        f"pool_regret={row['pool_best_feasible_regret']} "
                        f"r2_obj={row['oracle_feature_r2_objective_full']:.3f} "
                        f"r2_margin={row['oracle_feature_r2_margin_full']:.3f}",
                        flush=True,
                    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or f"lfos_oracle_gap_{time.strftime('%Y%m%d_%H%M%S')}"
    json_path = out_dir / f"{prefix}.json"
    csv_path = out_dir / f"{prefix}.csv"
    payload = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": float(time.time() - started),
        "config": vars(args),
        "rows": rows,
    }
    json_path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    flat_rows = []
    for row in rows:
        out = dict(row)
        out["source_counts"] = json.dumps(row["source_counts"], sort_keys=True)
        flat_rows.append(out)
    write_csv(csv_path, flat_rows)
    print(json.dumps(json_safe({"json": str(json_path), "csv": str(csv_path)}), indent=2))


if __name__ == "__main__":
    main()
