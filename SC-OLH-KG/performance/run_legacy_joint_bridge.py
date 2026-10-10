"""Connect the actual legacy frontend to source function priors and raw KG."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_archive import FrozenTransferArchive, resample_normalized_profiles
from core.designs import load_frozen_source_informed_design
from core.legacy_joint_kg import legacy_joint_utility, legacy_raw_joint_kg
from core.terminal_verification import verify_frozen_shortlist
from performance.benchmark_frozen_proposal_only import _select_shortlist
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.legacy_function_prior import legacy_public_coordinates, legacy_source_labels
from representation.orthogonal_function_prior import (
    OrthogonalLowOrderBasis, fit_orthogonal_function_prior, same_source_orthogonal_full_prior)
from representation.source_function_prior import head_batches

OUT = ROOT / "paper_artifacts/legacy_joint_bridge_v1"
SPEC = ROOT / "performance/manifests/legacy_joint_bridge_v1.json"


def function_moments(prior, psi):
    design = np.column_stack([np.ones(len(psi)), prior.basis.transform(psi)])
    return design @ prior.mean, design @ prior.covariance @ design.T + prior.discrepancy_variance * np.eye(len(psi))


def update_objective(mean, covariance, index, value, noise):
    column = covariance[:, index].copy()
    total = covariance[index, index] + noise
    mean += column * (value - mean[index]) / total
    covariance -= np.outer(column, column) / total


def frozen_support(primary, initial_rows, initial_points, observations):
    """Keep the legacy primary available to independent objective/safety comparison."""
    ordered = [tuple(primary)] + [tuple(row["point"]) for row in initial_rows]
    ordered += [initial_points[int(i)] for i in np.argsort(np.asarray(observations)[:, 1], kind="stable")]
    points = list(dict.fromkeys(ordered))[:3]
    old_primary = tuple(initial_rows[0]["point"])
    return [{"point": list(p), "shortlist_position": i + 1,
             "shortlist_role": "new_primary" if i == 0 else "retained_initial_support",
             "target_oracle_used": False, "verification_samples_used": False}
            for i, p in enumerate(points)], points.index(old_primary) + 1


def run():
    spec = json.loads(SPEC.read_text())
    started = time.process_time()
    rows, actual_search, actual_verification = [], 0, 0
    source_diagnostics = {}
    frozen_rows = []
    for domain in spec["domains"]:
        directory = OUT / "inputs" / domain
        archive = FrozenTransferArchive.load(directory / "archive.json")
        initial, design_contract = load_frozen_source_informed_design(directory / "design.json",
            heldout=domain, seed=spec["target_seed"], n0=spec["n0"], dimension=spec["target_dimension"])
        if archive.fingerprint != design_contract["source_archive_fingerprint"]:
            raise ValueError("frontend and function prior must use the same frozen source archive")
        problem = build_scalarized_problem(domain, spec["target_dimension"], spec["L"],
            spec["sigma"], spec["alpha"], spec["weights"])
        # Include every profile, without sorting/filtering by source responses.
        lifted = resample_normalized_profiles(np.vstack([t.X for t in archive.tasks]), problem.d)
        candidates = list(dict.fromkeys(list(initial) + [tuple(map(int, problem.continuous_to_int(x))) for x in lifted]))
        psi = legacy_public_coordinates(np.asarray(candidates) / problem.L)
        tasks = legacy_source_labels(archive)
        basis = OrthogonalLowOrderBasis(np.vstack([t["psi"] for t in tasks]))
        structural = [fit_orthogonal_function_prior(head_batches(tasks, h),
            [t["noise"][:, h] for t in tasks], basis,
            max_iterations=spec["source_max_iterations"]) for h in range(3)]
        full = [same_source_orthogonal_full_prior(p) for p in structural]
        source_calls = sum(t.simulator_calls for t in archive.tasks)
        if source_calls != spec["source_calls_per_reused_archive"] or any(t.X.shape[1] != spec["source_dimension"] for t in archive.tasks):
            raise ValueError("source archive does not match the declared pilot budget/dimension")
        source_diagnostics[domain] = {"candidate_count": len(candidates), "source_calls_reused": source_calls,
            "fitted_heads": [p.diagnostics for p in structural], "frontend_contract": design_contract}
        f_noise = float(np.mean(np.concatenate([t.replicate_variance[:, 0] for t in archive.tasks])))
        protocol = {"objective_reference": float(np.dot(problem.weights, problem.ref_point)),
                    "verification_replicates": spec["terminal_candidate_budgets"][0],
                    "chance_alpha": problem.alpha, "candidate_delta": spec["terminal_familywise_delta"] / 3}
        rng = np.random.default_rng(spec["target_seed"])
        observations = [problem.simulate(point, rng) for point in initial]
        actual_search += len(observations)
        old_rows, _ = _select_shortlist(problem, initial, observations)
        baseline = {"domain": domain, "arm": "legacy_proposal_only", "search_calls": 10,
                    "primary": tuple(old_rows[0]["point"]), "observations": np.asarray(observations).tolist(),
                    "initial_points_identical": True, "paid_eta_labels": 0}
        shortlist, incumbent = frozen_support(baseline["primary"], old_rows, initial, observations)
        frozen_rows.append((problem, baseline, shortlist, incumbent))
        for arm, priors in [("same_source_full", full), ("source_orthogonal", structural)]:
            f_mean, f_cov = function_moments(priors[0], psi)
            risk = JointFunctionRiskPosterior([priors[1]], [priors[2]], psi)
            arm_rng = np.random.default_rng(spec["target_seed"])
            # Advance the identical initial noise stream; use the already paid values.
            arm_rng.normal(size=3 * len(initial))
            indices, values, kg_records = [], [], []
            for index, value in enumerate(observations):
                risk.update_batch(index, float(value[1]) - problem.tau, None, 1)
                update_objective(f_mean, f_cov, index, float(value[0]), f_noise)
                indices.append(index); values.append(np.asarray(value).tolist())
            for calls in range(10, 14):
                mean_risk, covariance = risk.predict(full_cov=True)
                point_cov = covariance[np.arange(len(psi)), :, np.arange(len(psi)), :]
                utility, power = legacy_joint_utility(f_mean, np.diag(f_cov), mean_risk, point_cov, protocol)
                primary_index = int(utility.argmax())
                if calls in spec["record_stages"]:
                    row = {"domain": domain, "arm": arm, "search_calls": calls,
                           "primary": candidates[primary_index], "primary_index": primary_index,
                           "predicted_primary_utility": float(utility[primary_index]),
                           "predicted_primary_power": float(power[primary_index]),
                           "paid_candidate_indices": indices.copy(), "paid_observations": values.copy(),
                           "paid_eta_labels": 0, "risk_diagnostics": risk.diagnostics(),
                           "initial_points_identical": indices[:10] == list(range(10)),
                           "kg_steps": kg_records.copy()}
                    shortlist, incumbent = frozen_support(row["primary"], old_rows, initial, observations)
                    frozen_rows.append((problem, row, shortlist, incumbent))
                if calls == 13:
                    break
                noise = np.column_stack([np.full(len(psi), f_noise),
                    np.exp(mean_risk[:, 1] + .5 * point_cov[:, 1, 1])])
                scores = legacy_raw_joint_kg(f_mean, f_cov, mean_risk, covariance, noise, protocol,
                                            order=spec["kg_order"])
                index = int(scores.argmax())
                value = problem.simulate(candidates[index], arm_rng)
                actual_search += 1
                kg_records.append({"candidate": index, "score": float(scores[index]), "observed_heads": ["f", "g"]})
                risk.update_batch(index, float(value[1]) - problem.tau, None, 1)
                update_objective(f_mean, f_cov, index, float(value[0]), f_noise)
                indices.append(index); values.append(value.tolist())
            print(domain, arm, "search and recommendations frozen", flush=True)
    # All arms' search and shortlists finish before any terminal sample or truth.
    for problem, row, shortlist, incumbent in frozen_rows:
        deployed, verification = verify_frozen_shortlist(problem, shortlist, seed=spec["target_seed"],
            search_evaluation_count=row["search_calls"], candidate_budgets=spec["terminal_candidate_budgets"],
            familywise_delta=spec["terminal_familywise_delta"], method=spec["terminal_method"],
            objective_incumbent_position=incumbent,
            objective_comparison_budget=spec["terminal_objective_comparison_budget_per_policy"])
        actual_verification += verification["verification_budget"]
        row["verification"] = verification
        for label, point in [("primary", row["primary"]), ("deployed", deployed)]:
            g = problem.true_constraint_mean(point) - problem.tau
            sigma = float(problem.true_sigma(point)[1])
            f = problem.true_objective(point)
            score_protocol = {"objective_reference": float(np.dot(problem.weights, problem.ref_point)),
                "verification_replicates": spec["terminal_candidate_budgets"][0],
                "chance_alpha": problem.alpha, "candidate_delta": spec["terminal_familywise_delta"] / 3}
            utility, power = legacy_joint_utility(np.array([f]), np.zeros(1), np.array([[g, np.log(sigma ** 2)]]),
                np.zeros((1, 2, 2)), score_protocol)
            row[label + "_truth"] = {"objective": float(f), "chance_margin": float(g + norm.ppf(1 - problem.alpha) * sigma),
                "true_feasible": bool(g + norm.ppf(1 - problem.alpha) * sigma <= 0),
                "primary_verifier_power": float(power[0]), "primary_utility": float(utility[0])}
        row["deployed"] = list(deployed)
        rows.append(row)
    result = {"protocol": spec, "source_diagnostics": source_diagnostics, "rows": rows,
              "actual_calls": {"new_source": 0, "target_search": actual_search, "terminal": actual_verification},
              "cpu_seconds": time.process_time() - started}
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"actual_calls": result["actual_calls"], "cpu_seconds": result["cpu_seconds"],
        "rows": [{k: r[k] for k in ["domain", "arm", "search_calls", "primary_truth", "deployed_truth"]}
                 | {"certified": r["verification"]["certified"]} for r in rows]}, indent=2))
    return result


if __name__ == "__main__":
    run()
