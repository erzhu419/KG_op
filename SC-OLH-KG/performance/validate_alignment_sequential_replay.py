"""Sequential held-out replay for source-trained alignment admission.

The observation stream uses only frozen source boundary profiles and ordinary
target random samples.  No held-out target structural hook or oracle value is
used to construct or order the stream.  Target truth is read only on a
disjoint evaluation pool after each replay checkpoint.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.candidates import unique_candidates  # noqa: E402
from performance.benchmark_quality import (  # noqa: E402
    json_safe,
    parse_csv,
    parse_weights,
    write_csv,
)
from performance.validate_meta_spectral_basis import (  # noqa: E402
    build_problem,
    build_source_problems,
    coordinate_features,
    decision_quality,
    ridge_predict,
    target_rows,
)
from representation.meta_prior import (  # noqa: E402
    LearnedMetaPrior,
    PilotGatedMetaPriorBasis,
)


def fit_source_prior(args, heldout, seed):
    source_names = [
        name for name in parse_csv(args.domains) if name != heldout
    ]
    sources = build_source_problems(source_names, args, seed)
    prior = LearnedMetaPrior(
        local_dim=args.meta_local_dim,
        shared_dim=args.meta_shared_dim,
        component_stage="spectral",
        spectral_active_dim=args.active_dim,
        spectral_max_library_size=args.max_library_size,
        spectral_low_frequency_components=args.low_frequency_components,
        spectral_graph_neighbors=args.graph_neighbors,
        spectral_relevance_floor=args.relevance_floor,
        spectral_gate_boundary_weight=args.gate_boundary_weight,
        spectral_gate_dangerous_weight=args.gate_dangerous_weight,
        spectral_gate_selection_tolerance=args.gate_selection_tolerance,
        spectral_gate_calibration_quantile=args.gate_calibration_quantile,
        spectral_risk_alignment=True,
        spectral_alignment_active_dim=args.alignment_active_dim,
        spectral_alignment_subspace_dim=args.alignment_subspace_dim,
        spectral_alignment_domain_penalty=args.alignment_domain_penalty,
        spectral_alignment_target_ridge=args.alignment_target_ridge,
        spectral_alignment_target_min_gain=args.alignment_target_min_gain,
        spectral_alignment_target_min_bins=args.alignment_target_min_bins,
        spectral_alignment_source_episodes=args.alignment_source_episodes,
        spectral_alignment_admission=True,
        spectral_alignment_episode_pilot_size=(
            args.alignment_episode_pilot_size),
        spectral_alignment_episode_evaluation_size=(
            args.alignment_episode_evaluation_size),
        spectral_alignment_episode_ridge=args.alignment_episode_ridge,
        spectral_alignment_refit_interval=args.alignment_refit_interval,
        teacher_records_per_domain=args.source_boundary_records_per_domain,
        teacher_pool_size=args.source_boundary_pool_size,
        teacher_weight=args.source_boundary_weight,
        coordinate_mode=args.coordinate_mode,
        coordinate_relevance_floor=args.coordinate_relevance_floor,
        ridge=args.meta_ridge,
        seed=args.meta_seed + int(seed),
    ).fit_from_source_problems(
        sources,
        n_records_per_domain=args.source_records_per_domain,
        rng=np.random.default_rng(args.meta_seed + 1009 * int(seed)),
    )
    return prior, source_names


def replay_stream(prior, target, count, rng):
    source_profiles = prior.alignment_profile_candidates(
        target, n=max(2 * int(count), int(count) + 16), rng=rng)
    random = target_rows(target, max(int(count), 32), rng)
    rows = []
    sources = []
    profile_index = 0
    random_index = 0
    # A fixed 3:1 source-profile/random schedule is source-side policy only.
    # It does not inspect target responses while forming the replay stream.
    while len(rows) < int(count) and (
        profile_index < len(source_profiles) or random_index < len(random)
    ):
        use_profile = (
            len(rows) % 4 != 3 and profile_index < len(source_profiles)
        ) or random_index >= len(random)
        if use_profile:
            row = source_profiles[profile_index]
            source = "frozen_source_boundary_profile"
            profile_index += 1
        else:
            row = random[random_index]
            source = "target_random"
            random_index += 1
        row = tuple(int(value) for value in row)
        if row in rows:
            continue
        rows.append(row)
        sources.append(source)
    while len(rows) < int(count):
        row = tuple(int(value) for value in target.sample_random(rng))
        if row not in rows:
            rows.append(row)
            sources.append("target_random")
    return rows, sources, source_profiles


def evaluation_pool(prior, target, excluded, n, rng):
    excluded = set(excluded)
    profiles = prior.alignment_profile_candidates(
        target, n=max(int(n), len(prior.alignment_profile_templates)), rng=rng)
    random = target_rows(target, max(int(n), 64), rng)
    rows = []
    profile_rows = []
    for row in profiles:
        row = tuple(int(value) for value in row)
        if row not in excluded and row not in rows:
            rows.append(row)
            profile_rows.append(row)
    for row in random:
        row = tuple(int(value) for value in row)
        if row not in excluded and row not in rows:
            rows.append(row)
        if len(rows) >= int(n):
            break
    return rows[: int(n)], set(profile_rows)


def run_one(args, heldout, seed):
    prior, source_names = fit_source_prior(args, heldout, seed)
    target = build_problem(heldout, args)
    rng = np.random.default_rng(args.target_seed + 7919 * int(seed))
    checkpoints = sorted({
        int(value) for value in parse_csv(args.replay_checkpoints)
        if int(value) >= 6
    })
    maximum = max(checkpoints)
    stream, stream_sources, _ = replay_stream(prior, target, maximum, rng)
    observed = {}
    for row in stream:
        observed[row] = [np.asarray(target.simulate(row, rng), dtype=float)]
    test_rows, profile_test_rows = evaluation_pool(
        prior, target, stream, args.test_size, rng)
    truth = np.vstack([
        np.asarray(target.true_outputs(row), dtype=float) for row in test_rows
    ])
    coordinate_test = np.vstack([
        coordinate_features(prior, target, row) for row in test_rows
    ])
    rows = []
    for checkpoint in checkpoints:
        pilot_rows = stream[:checkpoint]
        pilot_observations = {
            row: observed[row] for row in pilot_rows
        }
        pilot_y = np.vstack([
            np.mean(np.asarray(pilot_observations[row]), axis=0)
            for row in pilot_rows
        ])
        coordinate_pilot = np.vstack([
            coordinate_features(prior, target, row) for row in pilot_rows
        ])
        objective_prediction = ridge_predict(
            coordinate_pilot,
            pilot_y[:, 0:1],
            coordinate_test,
            args.probe_ridge,
        )[:, 0]

        gate = PilotGatedMetaPriorBasis(
            prior, target, output_index=1, ridge=args.probe_ridge)
        selected = gate.fit_from_observations(
            pilot_observations, output_index=1)
        gate_diag = gate.diagnostics()
        stage1 = gate_diag.get("stage1_selected_basis", "coordinate")

        def predict_constraint(variant):
            pilot_features = np.vstack([
                gate._variant_features(row, variant) for row in pilot_rows
            ])
            test_features = np.vstack([
                gate._variant_features(row, variant) for row in test_rows
            ])
            return ridge_predict(
                pilot_features,
                pilot_y[:, 1:2],
                test_features,
                args.probe_ridge,
            )[:, 0]

        stage1_constraint = predict_constraint(stage1)
        frozen_constraint = (
            predict_constraint("frozen_risk_aligned_coordinate")
            + gate._risk_alignment_source_guard()
        )
        deployed_constraint = (
            frozen_constraint
            if selected == "frozen_risk_aligned_coordinate"
            else stage1_constraint
        )
        stage1_prediction = np.column_stack([
            objective_prediction, stage1_constraint])
        frozen_prediction = np.column_stack([
            objective_prediction, frozen_constraint])
        deployed_prediction = np.column_stack([
            objective_prediction, deployed_constraint])
        stage1_quality = decision_quality(
            target, test_rows, truth, stage1_prediction)
        frozen_quality = decision_quality(
            target, test_rows, truth, frozen_prediction)
        deployed_quality = decision_quality(
            target, test_rows, truth, deployed_prediction)
        risk_diag = gate_diag.get("risk_alignment", {})
        support = risk_diag.get("boundary_support", {})
        admission = risk_diag.get("source_episode_admission", {})
        true_pilot_feasible = np.asarray([
            target.is_truly_feasible(row) for row in pilot_rows
        ], dtype=bool)
        test_true_feasible = np.asarray([
            target.is_truly_feasible(row) for row in test_rows
        ], dtype=bool)
        rows.append({
            "heldout": heldout,
            "seed": int(seed),
            "checkpoint": int(checkpoint),
            "source_domains": ",".join(source_names),
            "source_episode_status": prior.alignment_episode_diagnostics.get(
                "status", "missing"),
            "source_episode_count": int(
                prior.alignment_episode_diagnostics.get("n_episodes", 0)),
            "source_episode_win_rate": prior.alignment_episode_diagnostics.get(
                "evaluation_win_rate"),
            "source_episode_median_gain": prior.alignment_episode_diagnostics.get(
                "median_evaluation_gain"),
            "source_profile_template_count": int(
                len(prior.alignment_profile_templates)),
            "stream_profile_count": int(sum(
                value == "frozen_source_boundary_profile"
                for value in stream_sources[:checkpoint]
            )),
            "stream_random_count": int(sum(
                value == "target_random"
                for value in stream_sources[:checkpoint]
            )),
            "target_structural_hooks_used": False,
            "target_oracle_used_for_stream": False,
            "target_truth_used_for_evaluation_only": True,
            "observed_boundary_feasible": int(support.get("n_feasible", 0)),
            "observed_boundary_infeasible": int(support.get("n_infeasible", 0)),
            "true_pilot_feasible_count_diagnostic": int(np.sum(
                true_pilot_feasible)),
            "selected_basis": str(selected),
            "stage1_basis": str(stage1),
            "alignment_enabled": bool(
                selected == "frozen_risk_aligned_coordinate"),
            "alignment_rejection_reasons": "|".join(
                risk_diag.get("rejection_reasons", [])),
            "source_admission_status": admission.get("status"),
            "source_admission_reasons": "|".join(
                admission.get("rejection_reasons", [])),
            "source_predicted_gain": admission.get(
                "predicted_evaluation_gain"),
            "source_gain_lower_quartile": admission.get(
                "source_gain_lower_quartile"),
            "source_local_win_rate": admission.get(
                "source_local_win_rate"),
            "stage1_decision_loss": stage1_quality["total"],
            "frozen_alignment_decision_loss": frozen_quality["total"],
            "deployed_decision_loss": deployed_quality["total"],
            "frozen_over_stage1_gain": (
                stage1_quality["total"] - frozen_quality["total"]),
            "deployed_over_stage1_gain": (
                stage1_quality["total"] - deployed_quality["total"]),
            "stage1_false_feasible_rate": stage1_quality[
                "false_feasible_rate"],
            "frozen_false_feasible_rate": frozen_quality[
                "false_feasible_rate"],
            "deployed_false_feasible_rate": deployed_quality[
                "false_feasible_rate"],
            "test_true_feasible_rate": float(np.mean(test_true_feasible)),
            "test_source_profile_fraction": float(np.mean([
                row in profile_test_rows for row in test_rows
            ])),
        })
    return rows


def _run_one_task(task):
    return run_one(*task)


def summarize(rows):
    by_cell = {}
    for heldout in sorted({row["heldout"] for row in rows}):
        for checkpoint in sorted({
            row["checkpoint"] for row in rows if row["heldout"] == heldout
        }):
            cell = [
                row for row in rows
                if row["heldout"] == heldout
                and row["checkpoint"] == checkpoint
            ]
            gains = np.asarray([
                row["deployed_over_stage1_gain"] for row in cell], dtype=float)
            challenger = np.asarray([
                row["frozen_over_stage1_gain"] for row in cell], dtype=float)
            by_cell[f"{heldout}:N{checkpoint}"] = {
                "n_runs": int(len(cell)),
                "alignment_activation_rate": float(np.mean([
                    row["alignment_enabled"] for row in cell
                ])),
                "median_deployed_gain": float(np.median(gains)),
                "deployed_win_rate": float(np.mean(gains > 0.0)),
                "median_frozen_challenger_gain": float(np.median(challenger)),
                "frozen_challenger_win_rate": float(np.mean(challenger > 0.0)),
                "median_deployed_false_feasible_rate": float(np.median([
                    row["deployed_false_feasible_rate"] for row in cell
                ])),
                "median_stage1_false_feasible_rate": float(np.median([
                    row["stage1_false_feasible_rate"] for row in cell
                ])),
            }
    activated = [row for row in rows if row["alignment_enabled"]]
    by_domain_gain = {
        heldout: float(np.median([
            row["deployed_over_stage1_gain"] for row in activated
            if row["heldout"] == heldout
        ]))
        for heldout in sorted({row["heldout"] for row in activated})
    }
    accepted = bool(
        activated
        and np.median([
            row["deployed_over_stage1_gain"] for row in activated
        ]) > 0.0
        and np.mean([
            row["deployed_over_stage1_gain"] > 0.0 for row in activated
        ]) >= 0.60
        and all(value >= -0.05 for value in by_domain_gain.values())
        and np.median([
            row["deployed_false_feasible_rate"] for row in activated
        ]) <= np.median([
            row["stage1_false_feasible_rate"] for row in activated
        ])
    )
    return {
        "accepted_for_kg_gate": accepted,
        "criterion": (
            "activated replay median gain > 0, win rate >= 0.60, every "
            "activated held-out median >= -0.05, and false-feasible no worse"
        ),
        "n_rows": int(len(rows)),
        "n_activated": int(len(activated)),
        "activation_rate": float(len(activated) / max(len(rows), 1)),
        "median_activated_gain": (
            float(np.median([
                row["deployed_over_stage1_gain"] for row in activated
            ])) if activated else None
        ),
        "activated_win_rate": (
            float(np.mean([
                row["deployed_over_stage1_gain"] > 0.0 for row in activated
            ])) if activated else None
        ),
        "by_activated_domain_median_gain": by_domain_gain,
        "by_cell": by_cell,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domains",
        default=(
            "FactorShockStatePolicyRZDT1,InventorySupplyChain,"
            "QueueResourceControl,HighDimStatePolicyRZDT1,StatePolicyRZDT1"
        ),
    )
    parser.add_argument(
        "--heldouts",
        default=(
            "FactorShockStatePolicyRZDT1,InventorySupplyChain,"
            "QueueResourceControl"
        ),
    )
    parser.add_argument("--d", type=int, default=50)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--source_records_per_domain", type=int, default=64)
    parser.add_argument("--source_augments", type=int, default=1)
    parser.add_argument("--source_sigma_jitter", type=float, default=0.20)
    parser.add_argument("--source_alpha_jitter", type=float, default=0.25)
    parser.add_argument("--source_weight_jitter", type=float, default=0.05)
    parser.add_argument(
        "--source_boundary_records_per_domain", type=int, default=32)
    parser.add_argument("--source_boundary_pool_size", type=int, default=512)
    parser.add_argument("--source_boundary_weight", type=float, default=3.0)
    parser.add_argument("--replay_checkpoints", default="10,15,20,30,40")
    parser.add_argument("--test_size", type=int, default=128)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--meta_local_dim", type=int, default=3)
    parser.add_argument("--meta_shared_dim", type=int, default=3)
    parser.add_argument("--active_dim", type=int, default=6)
    parser.add_argument("--max_library_size", type=int, default=64)
    parser.add_argument("--low_frequency_components", type=int, default=8)
    parser.add_argument("--graph_neighbors", type=int, default=10)
    parser.add_argument("--relevance_floor", type=float, default=0.05)
    parser.add_argument("--gate_boundary_weight", type=float, default=2.0)
    parser.add_argument("--gate_dangerous_weight", type=float, default=3.0)
    parser.add_argument("--gate_selection_tolerance", type=float, default=0.02)
    parser.add_argument("--gate_calibration_quantile", type=float, default=0.90)
    parser.add_argument("--alignment_active_dim", type=int, default=4)
    parser.add_argument("--alignment_subspace_dim", type=int, default=2)
    parser.add_argument("--alignment_domain_penalty", type=float, default=0.5)
    parser.add_argument("--alignment_target_ridge", type=float, default=5.0)
    parser.add_argument("--alignment_target_min_gain", type=float, default=0.02)
    parser.add_argument("--alignment_target_min_bins", type=int, default=3)
    parser.add_argument("--alignment_refit_interval", type=int, default=5)
    parser.add_argument("--alignment_source_episodes", type=int, default=6)
    parser.add_argument("--alignment_episode_pilot_size", type=int, default=10)
    parser.add_argument(
        "--alignment_episode_evaluation_size", type=int, default=24)
    parser.add_argument("--alignment_episode_ridge", type=float, default=0.1)
    parser.add_argument(
        "--coordinate_mode",
        choices=["pca", "stable_supervised"],
        default="stable_supervised",
    )
    parser.add_argument("--coordinate_relevance_floor", type=float, default=0.05)
    parser.add_argument("--meta_ridge", type=float, default=1e-4)
    parser.add_argument("--probe_ridge", type=float, default=0.1)
    parser.add_argument("--meta_seed", type=int, default=20260710)
    parser.add_argument("--target_seed", type=int, default=20260711)
    parser.add_argument("--out_dir", default=str(ROOT / "profiles"))
    parser.add_argument(
        "--out_prefix", default="alignment_sequential_replay")
    args = parser.parse_args()

    tasks = [
        (args, heldout, seed)
        for heldout in parse_csv(args.heldouts)
        for seed in range(args.seed_start, args.seed_start + args.n_seeds)
    ]
    if int(args.jobs) > 1:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as executor:
            groups = list(executor.map(
                _run_one_task,
                tasks,
            ))
        rows = [row for group in groups for row in group]
    else:
        rows = [
            row
            for _, heldout, seed in tasks
            for row in run_one(args, heldout, seed)
        ]
    summary = summarize(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{args.out_prefix}_rows.csv"
    json_path = out_dir / f"{args.out_prefix}.json"
    write_csv(rows_path, rows)
    json_path.write_text(json.dumps(json_safe({
        "config": vars(args),
        "summary": summary,
        "rows": rows,
    }), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(json_safe({
        "summary": summary,
        "rows_csv": str(rows_path),
        "json": str(json_path),
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
