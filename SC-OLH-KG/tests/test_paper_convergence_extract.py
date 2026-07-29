import hashlib
import json

from core.designs import integer_design_fingerprint
from performance.paper_convergence_extract import build_convergence
from problems.rzdt import make_problem
from problems.single_objective import ScalarizedProblem


def _contract(domain, dimension, *, shared_shock_scale=None):
    payload = {
        "domain": domain,
        "dimension": dimension,
        "L": 100,
        "sigma": 0.04,
        "alpha": 0.05,
        "scalarization_weights": [0.5, 0.5],
        "heteroscedastic": True,
        "tau": 0.0,
        "shared_shock_scale": shared_shock_scale,
        "task_geometry": "nominal",
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()
    return payload, hashlib.sha256(encoded).hexdigest()


def _problem(domain, dimension, *, shared_shock_scale=None):
    kwargs = {}
    if shared_shock_scale is not None:
        kwargs["shared_shock_scale"] = shared_shock_scale
    return ScalarizedProblem(make_problem(
        domain,
        d=dimension,
        L=100,
        sigma=0.04,
        alpha=0.05,
        **kwargs,
    ))


def _truth(problem, point):
    import statistics

    objective = problem.true_objective(point)
    mean = problem.true_constraint_mean(point)
    sigma = problem.true_sigma(point)[1]
    margin = mean + statistics.NormalDist().inv_cdf(
        1.0 - problem.alpha) * sigma
    return float(objective), float(margin), bool(margin <= 0.0)


def _record(path, contract, fingerprint, *, method, calls):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "track_id": "final_frozen_source_frontend_backend_d1000_n13",
        "path": str(path),
        "result_sha256": digest,
        "status": "ok",
        "domain": contract["domain"],
        "seed": 80,
        "target_dimension": contract["dimension"],
        "problem_contract": contract,
        "problem_contract_fingerprint": fingerprint,
        "method_identity": method,
        "target_search_calls": calls,
    }


def test_reconstructs_history_without_exporting_policy_vectors(tmp_path):
    domain = "InventorySupplyChain"
    dimension = 6
    problem = _problem(domain, dimension)
    points = [
        [58, 58, 36, 36, 42, 42],
        [10, 10, 10, 10, 10, 10],
        [62, 62, 38, 38, 50, 50],
    ]
    objective, margin, feasible = _truth(problem, points[0])
    payload = {
        "status": "ok",
        "result": {
            "method": "botorch_saasbo",
            "history": [{"x": point, "y": [0.0, 0.0]}
                        for point in points],
            "n_search_simulations": 3,
            "x_recommended": points[0],
            "true_objective": objective,
            "true_chance_margin": margin,
            "true_feasible": feasible,
            "feasible_regret": 0.05,
        },
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    contract, fingerprint = _contract(domain, dimension)
    audit = {"records": [_record(
        path,
        contract,
        fingerprint,
        method="canonical_saasbo_every_iteration",
        calls=3,
    )]}

    rows, manifest = build_convergence(audit)

    assert manifest["status"] == "complete"
    assert manifest["trace_row_count"] == 3
    assert manifest["terminal_validation_failure_count"] == 0
    assert [row["target_call"] for row in rows] == [1, 2, 3]
    assert all("point" not in row and "x" not in row for row in rows)
    assert all(row["verification_samples_included"] is False
               for row in rows)
    assert rows[0]["point_fingerprint"] == integer_design_fingerprint([
        points[0]])


def test_reconstructs_sc_initial_design_and_post_run_action(tmp_path):
    domain = "QueueResourceControl"
    dimension = 6
    problem = _problem(domain, dimension)
    initial = [
        [64, 64, 38, 38, 52, 52],
        [20, 20, 20, 20, 20, 20],
    ]
    action = [68, 68, 40, 40, 50, 50]
    action_objective, action_margin, action_feasible = _truth(
        problem, action)
    objective, margin, feasible = _truth(problem, initial[0])
    payload = {
        "config": {
            "d": dimension,
            "n0": 2,
            "initial_design_points": initial,
        },
        "rows": [{
            "n0": 2,
            "n_search_simulations": 3,
            "online_action_trace": [{
                "target_call": 3,
                "action_kind": "new",
                "x_fingerprint": integer_design_fingerprint([action]),
                "true_objective_post_run": action_objective,
                "true_chance_margin_post_run": action_margin,
                "true_feasible_post_run": action_feasible,
                "target_oracle_used_for_decision": False,
            }],
            "x_recommended": initial[0],
            "true_objective": objective,
            "true_chance_margin": margin,
            "true_feasible": feasible,
            "feasible_regret": 0.04,
        }],
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    contract, fingerprint = _contract(domain, dimension)
    audit = {"records": [_record(
        path,
        contract,
        fingerprint,
        method="scolh:v69_feasible_first_verified_initial_incumbent",
        calls=3,
    )]}

    rows, manifest = build_convergence(audit)

    assert manifest["status"] == "complete"
    assert [row["phase"] for row in rows] == [
        "initial_design", "initial_design", "adaptive_search"]
    assert rows[-1]["point_fingerprint"] == integer_design_fingerprint([
        action])


def test_reconstructs_frozen_proposal_truth_audit(tmp_path):
    domain = "InventorySupplyChain"
    dimension = 6
    problem = _problem(domain, dimension)
    points = [
        [58, 58, 36, 36, 42, 42],
        [62, 62, 38, 38, 50, 50],
    ]
    truths = []
    for point in points:
        objective, margin, feasible = _truth(problem, point)
        truths.append({
            "x_recommended": point,
            "true_objective": objective,
            "true_chance_margin": margin,
            "true_feasible": feasible,
        })
    payload = {
        "status": "ok",
        "method": "frozen_crossdim_proposal_only",
        "information_contract": {
            "frozen_initial_points": points,
        },
        "n_search_simulations": 2,
        "x_recommended": points[0],
        **truths[0],
        "feasible_regret": 0.05,
        "initial_truth_audit": {
            "computed_after_shortlist_freeze_and_verification": True,
            "used_for_selection_or_certification": False,
            "rows": truths,
        },
    }
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    contract, fingerprint = _contract(domain, dimension)
    audit = {"records": [_record(
        path,
        contract,
        fingerprint,
        method="frozen_crossdim_proposal_only",
        calls=2,
    )]}

    rows, manifest = build_convergence(audit)

    assert manifest["status"] == "complete"
    assert manifest["trace_row_count"] == 2
    assert all(row["phase"] == "initial_design" for row in rows)
    assert all(
        row["target_truth_used_post_run_only"] is True for row in rows)


def test_problem_contract_mismatch_fails_terminal_validation(tmp_path):
    domain = "FactorShockStatePolicyRZDT1"
    dimension = 6
    scale_zero_problem = _problem(
        domain, dimension, shared_shock_scale=0.0)
    point = [25, 75, 75, 75, 75, 75]
    objective, margin, feasible = _truth(scale_zero_problem, point)
    payload = {
        "status": "ok",
        "result": {
            "history": [{"x": point, "y": [0.0, 0.0]}],
            "n_search_simulations": 1,
            "x_recommended": point,
            "true_objective": objective,
            "true_chance_margin": margin,
            "true_feasible": feasible,
            "feasible_regret": 0.0,
        },
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    scale_one_contract, fingerprint = _contract(
        domain, dimension, shared_shock_scale=1.0)
    audit = {"records": [_record(
        path,
        scale_one_contract,
        fingerprint,
        method="mismatched",
        calls=1,
    )]}

    _, manifest = build_convergence(audit)

    assert manifest["status"] == "incomplete"
    assert manifest["terminal_validation_failure_count"] == 1
