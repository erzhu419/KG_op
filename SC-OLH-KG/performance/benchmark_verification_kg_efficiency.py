"""Local, single-thread kernel comparison using existing initial/logged posteriors."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.run_meta_prior_mechanism import observe, task_truth
from performance.run_meta_prior_mechanism_v2 import prepare_sources
from performance.run_meta_prior_total_budget import fit_priors, simulator_supports
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior, head_batches
from representation.transfer_mixture import TransferMixtureGPR

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_kg_efficiency_20261003"


def initial_models(bank):
    protocol = json.loads((ROOT / "performance/meta_prior_candidate384_protocol.json").read_text())
    psi = np.random.default_rng(51040).uniform(-2, 2, (bank, 8))
    priors, _ = fit_priors(40, "shared_structure", "fixed_source", psi[:96], protocol)
    models = [p.target_model(psi) for p in priors]
    rng = np.random.default_rng(51040)
    rng.uniform(-2, 2, (96, 8))
    queries = rng.choice(96, 8, replace=False)
    truth = task_truth(psi, simulator_supports(40, "shared_structure", 8), 65000)
    truth[:, 1] += 1
    for i in queries:
        y, noise = observe(truth[i], 65000, int(i), 0, 12)
        for h, model in enumerate(models):
            model.update([int(i)], y[h], noise[h])
    return models, np.arange(bank)[:, None], protocol


def logged_models(seed):
    protocol = json.loads((ROOT / "performance/meta_prior_total_budget_protocol.json").read_text())
    protocol.update(protocol["allocations"]["n24_v928"])
    cases = json.loads((ROOT / "paper_artifacts/meta_prior_total_budget_v6_20261002/failure_cases.json").read_text())
    case = next(c for c in cases if c["row"]["seed"] == seed
                and c["row"]["arm"] == "risk_ard_mixture" and c["row"]["allocation"] == "n24_v928")
    row = case["row"]
    psi = np.random.default_rng(51000 + seed).uniform(-2, 2, (384, 8))
    reference = psi[:96]
    tasks, supports = prepare_sources(seed, row["regime"], protocol)
    batches = [head_batches(tasks, h) for h in range(3)]
    noises = [[t["noise"][:, h] for t in tasks] for h in range(3)]
    full = [SourceFunctionPrior.fit(batches[h], noises[h], reference, "fixed_source") for h in range(3)]
    structural = [SourceFunctionPrior.from_basis(batches[h], noises[h],
        FixedFunctionBasis(reference, case["source_info"][h]["selected_indices"]), "conditional_hard") for h in range(2)]
    tau = np.asarray(case["source_info"][2]["source_function_variance"])
    signal = np.mean([max(float(np.var(b.signals) - np.mean(v)), .02) for b, v in zip(batches[2], noises[2])])
    mean = np.r_[np.mean([np.mean(b.signals) for b in batches[2]]), np.zeros(68)]
    structural.append(SourceFunctionPrior(FixedFunctionBasis(reference), mean,
        np.diag(np.r_[.25, tau + .01 * signal / 68]), .02, {}))
    models = [TransferMixtureGPR([structural[h], full[h]], psi, reference) for h in range(3)]
    target_seed = 61000 + 100 * seed + protocol["regimes"].index(row["regime"])
    truth = task_truth(psi, supports, target_seed)
    truth[:, 1] += 1
    visits = np.zeros(384, dtype=int)
    for i in row["queries"]:
        y, noise = observe(truth[i], target_seed, i, int(visits[i]), 12)
        for h, model in enumerate(models):
            model.update([i], y[h], noise[h])
        visits[i] += 1
    for model, saved in zip(models, row["final_model_state"]):
        np.testing.assert_allclose(model.diagnostics()["structural_weight"], saved["structural_weight"], atol=1e-10)
    return models, np.arange(384)[:, None], protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["initial96", "initial384", "logged52", "logged61"], required=True)
    parser.add_argument("--implementation", choices=["reference", "factorized"], required=True)
    args = parser.parse_args()
    if args.implementation == "reference":
        spec = importlib.util.spec_from_file_location("reference_verification_kg", ARTIFACTS / "reference_verification_kg.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        import core.verification_kg as module
    models, candidates, protocol = (initial_models(int(args.case[7:])) if args.case.startswith("initial")
                                    else logged_models(int(args.case[6:])))
    # Warm the same posterior caches and reset Linux peak-RSS after setup.
    for model in models:
        model.posterior_mean_many(candidates)
        model.posterior_var_many(candidates)
    states = [(model.a.copy(), model.C.copy()) for model in models]
    Path("/proc/self/clear_refs").write_text("5\n")
    start = time.perf_counter()
    scores = module.verification_kg_scores(models, candidates, protocol)
    elapsed = time.perf_counter() - start
    for model, (a, covariance) in zip(models, states):
        np.testing.assert_array_equal(model.a, a)
        np.testing.assert_array_equal(model.C, covariance)
    peak_kb = int(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines()
                       if line.startswith("VmHWM:")))
    result = {"case": args.case, "implementation": args.implementation,
        "bank": len(candidates), "verification_replicates": protocol["verification_replicates"],
        "seconds": elapsed, "peak_rss_mb": peak_kb / 1024,
        "action": int(np.argmax(scores)), "scores": scores.tolist(), "posterior_unchanged": True}
    (ARTIFACTS / f"{args.case}_{args.implementation}.json").write_text(json.dumps(result, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "scores"}), flush=True)


if __name__ == "__main__":
    main()
