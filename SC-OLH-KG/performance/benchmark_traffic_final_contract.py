#!/usr/bin/env python3
"""Run the frozen paper front end with an audited BoTorch SUMO backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
GPR_KG_CODE = REPO_ROOT / "Final_Submission" / "GPR_KG_Code"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GPR_KG_CODE))

from baselines.botorch_adapters import (  # noqa: E402
    BoTorchBaseline,
    BoTorchBaselineConfig,
    botorch_runtime_fingerprint,
)
from core.designs import load_frozen_source_informed_design  # noqa: E402
from core.terminal_verification import (  # noqa: E402
    freeze_objective_incumbent_shortlist,
    select_initial_empirical_objective_incumbent,
)
from performance.benchmark_quality import json_safe  # noqa: E402
from performance.execution_provenance import (  # noqa: E402
    attach_execution_provenance,
)
from performance.materialize_external_traffic_design import (  # noqa: E402
    TARGET_DOMAIN,
)
from performance.paper_method_contract import (  # noqa: E402
    TARGET_N0,
    TARGET_SEARCH_CALLS,
    validate_frozen_proposal_payload,
)
from problems.traffic_ingolstadt21 import (  # noqa: E402
    Ingolstadt21ScalarizedTrafficProblem,
)


EXTERNAL_VERIFIER_CONTRACT = (
    "fresh_seed_familywise_exact_binomial_shortlist_v1")

BACKEND_CONTRACTS = {
    "botorch_saasbo": "canonical_botorch_saasbo_every_iteration_v1",
    "botorch_scbo": "official_botorch_scbo_cpu_external_v1",
    "botorch_turbo": "official_botorch_turbo_cpu_external_v1",
}


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_design(path, *, seed, n0, dimension):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    audit = validate_frozen_proposal_payload(payload, expected_n0=n0)
    points, contract = load_frozen_source_informed_design(
        path,
        heldout=TARGET_DOMAIN,
        seed=seed,
        n0=n0,
        dimension=dimension,
    )
    if payload.get("sumo_simulator_calls_during_materialization") != 0:
        raise ValueError("traffic proposal materialization evaluated SUMO")
    return payload, points, {**contract, **audit}


def run_one(args):
    if int(args.n0) != TARGET_N0:
        raise ValueError("traffic front-end contract requires n0=10")
    if int(args.N) < int(args.n0):
        raise ValueError("traffic search budget N must be at least n0")
    backend = str(args.backend).strip().lower()
    if backend not in BACKEND_CONTRACTS:
        raise ValueError(f"unsupported traffic backend {backend!r}")
    problem = Ingolstadt21ScalarizedTrafficProblem(
        weights=(float(args.weight_f1), float(args.weight_f2)),
        seed=int(args.seed),
        true_replications=1,
        sigma_replications=2,
        historical_anchor_policy="strict_none",
    )
    design_payload, initial_points, design_contract = _load_design(
        args.initial_design_file,
        seed=int(args.seed),
        n0=int(args.n0),
        dimension=int(problem.d),
    )
    method_label = str(args.method_label)
    partition = str(args.partition_method)
    if not method_label or not partition:
        raise ValueError("paper traffic method and partition labels are required")
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    result_path = output_dir / "result.json"
    if args.resume and summary_path.exists() and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    checkpoint_path = (
        Path(args.checkpoint_dir) / f"seed{int(args.seed):04d}.pkl")
    config = BoTorchBaselineConfig(
        N=int(args.N),
        n0=int(args.n0),
        seed=int(args.seed),
        method=backend,
        raw_samples=int(args.raw_samples),
        num_restarts=int(args.num_restarts),
        maxiter=int(args.maxiter),
        ts_candidates=int(args.ts_candidates),
        timeout_sec=float(args.candidate_timeout_sec),
        certification_beta=float(args.beta_g),
        saas_warmup_steps=int(args.saas_warmup_steps),
        saas_num_samples=int(args.saas_num_samples),
        saas_thinning=int(args.saas_thinning),
        saas_max_tree_depth=int(args.saas_max_tree_depth),
        saas_mc_samples=int(args.saas_mc_samples),
        saas_constrained=True,
        strict_failures=True,
        saas_fallback_after_failures=False,
        use_problem_initial_samples=False,
        use_boundary_initial_samples=False,
        initial_design="frozen_source_informed",
        initial_points=initial_points,
        checkpoint_path=str(checkpoint_path),
        checkpoint_resume=True,
        checkpoint_interval=1,
        progress_logging=True,
        progress_label=f"paper-traffic:seed={int(args.seed)}",
        torch_device=str(args.torch_device),
        saas_parallel_models=bool(backend == "botorch_saasbo"),
        saas_parallel_min_total_steps=64,
        saas_parallel_threads_per_model=int(
            args.saas_parallel_threads_per_model),
        saas_refit_schedule="every_iteration",
        torch_deterministic=bool(args.torch_deterministic),
    )
    started = time.time()
    optimizer = BoTorchBaseline(problem, config)
    result = optimizer.run(
        freeze_terminal_shortlist=True,
        evaluate_truth=False,
        terminal_probability_slack=0.05,
        terminal_require_provider=True,
        terminal_shortlist_mode=(
            "posterior_objective_challenger_then_safe"),
        terminal_shortlist_size=3,
        terminal_maximum_violation_probability=0.5,
    )
    initial_history = optimizer.history[: int(args.n0)]
    incumbent = select_initial_empirical_objective_incumbent(
        [point for point, _ in initial_history],
        [observation for _, observation in initial_history],
        n0=int(args.n0),
    )
    shortlist, incumbent_audit = freeze_objective_incumbent_shortlist(
        result["frozen_terminal_shortlist"],
        incumbent,
        shortlist_size=3,
    )
    result.update({
        "status": "ok",
        "frozen_terminal_shortlist": shortlist,
        "terminal_objective_incumbent": incumbent_audit,
        "search_calls": int(args.N),
        "verification_calls": 0,
        "source_calls": int(
            design_payload["source_archive_simulator_calls"]),
        "target_oracle_used": False,
        "external_verification_pending": True,
        "external_verifier_contract": EXTERNAL_VERIFIER_CONTRACT,
        "paper_frontend_contract_id": design_contract["contract_id"],
        "paper_backend_contract_id": BACKEND_CONTRACTS[backend],
        "backend": backend,
        "canonical_final_backend": bool(
            backend == "botorch_saasbo"
            and int(args.N) == TARGET_SEARCH_CALLS),
        "source_selection_mode": design_payload["source_selection_mode"],
        "initial_design_contract": design_contract,
        "runtime": botorch_runtime_fingerprint(str(args.torch_device)),
        "wall_time_sec": float(time.time() - started),
    })
    attach_execution_provenance(result)
    summary = {
        "schema_version": 1,
        "method": method_label,
        "backend": backend,
        "partition_method": partition,
        "problem": TARGET_DOMAIN,
        "N": int(args.N),
        "n0": int(args.n0),
        "seed": int(args.seed),
        "d": int(problem.d),
        "tau": float(problem.tau),
        "alpha": float(problem.alpha),
        "weights": [float(args.weight_f1), float(args.weight_f2)],
        "final_pareto_set": [
            list(map(int, row["point"])) for row in shortlist
        ],
        "final_recommendation": list(map(
            int, shortlist[0]["point"])),
        "frozen_terminal_shortlist": shortlist,
        "search_result": result,
        "information_contract": {
            "source_calls": int(
                design_payload["source_archive_simulator_calls"]),
            "target_search_calls": int(args.N),
            "target_verification_calls": 0,
            "target_truth_diagnostics_during_search": 0,
            "target_oracle_used": False,
            "historical_traffic_anchor_used": False,
            "sumo_calls_used_to_fit_proposal": 0,
            "proposal_frozen_before_target_observations": True,
            "external_verification_pending": True,
        },
        "execution_provenance": result["execution_provenance"],
    }
    _atomic_json(summary_path, summary)
    _atomic_json(result_path, result)
    print(json.dumps({
        "status": "ok",
        "summary": str(summary_path),
        "result": str(result_path),
        "seed": int(args.seed),
        "search_calls": int(args.N),
        "shortlist_size": len(shortlist),
        "truth_metrics_evaluated": False,
    }, indent=2), flush=True)
    print("DONE", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-design-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--partition-method", required=True)
    parser.add_argument(
        "--backend",
        choices=tuple(BACKEND_CONTRACTS),
        default="botorch_saasbo",
    )
    parser.add_argument("--N", type=int, default=13)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--weight-f1", type=float, default=0.5)
    parser.add_argument("--weight-f2", type=float, default=0.5)
    parser.add_argument("--raw-samples", type=int, default=1024)
    parser.add_argument("--num-restarts", type=int, default=10)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--candidate-timeout-sec", type=float, default=3600.0)
    parser.add_argument("--ts-candidates", type=int, default=2000)
    parser.add_argument("--beta-g", type=float, default=2.0)
    parser.add_argument("--saas-warmup-steps", type=int, default=256)
    parser.add_argument("--saas-num-samples", type=int, default=128)
    parser.add_argument("--saas-thinning", type=int, default=16)
    parser.add_argument("--saas-max-tree-depth", type=int, default=6)
    parser.add_argument("--saas-mc-samples", type=int, default=256)
    parser.add_argument("--saas-parallel-threads-per-model", type=int, default=6)
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--torch-deterministic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_one(args)


if __name__ == "__main__":
    main()
