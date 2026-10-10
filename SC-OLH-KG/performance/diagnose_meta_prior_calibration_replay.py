"""Post-score the failed replay branch using already reconstructed records."""
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.verification_kg import predictive_pass_probability
from performance.replay_meta_prior_calibration import ARTIFACTS, restore_context
from performance.run_meta_prior_mechanism import observe
from performance.run_meta_prior_total_budget import simulator_supports
from representation.transfer_mixture import TransferMixtureGPR


def main():
    summary = json.loads((ARTIFACTS / "summary.json").read_text())
    execution = json.loads((ARTIFACTS / "execution_plan.json").read_text())
    spec, frozen = execution["stage_a"], execution["base_protocol"]
    protocol = {**frozen, **frozen["allocations"][spec["allocation"]]}
    records = json.loads(gzip.decompress((ARTIFACTS / "inputs.json.gz").read_bytes()))["records"]
    choices = [(52, "shared_structure", "sample_variance_plugin"),
               (61, "separate_risk_support", "pre_batch_risk_predictive"),
               (62, "high_frequency_risk", "pre_batch_risk_predictive"),
               (60, "high_frequency_risk", "pre_batch_risk_predictive")]
    diagnostics = []
    for seed, regime, update in choices:
        scored = next(r for r in summary["rows"] if (r["seed"], r["regime"], r["arm"], r["update"]) ==
                      (seed, regime, "risk_ard_mixture", update))
        record = next(r for r in records if (r["row"]["seed"], r["row"]["regime"], r["row"]["arm"]) ==
                      (seed, regime, "risk_ard_mixture"))
        mean, variance, truth = (np.array(scored[k]) for k in ("posterior_mean", "posterior_variance", "truth"))
        replacement = {}
        for head, name in ((1, "true_g_mean"), (2, "true_eta_mean")):
            m = mean.copy()
            m[head] = truth[head]
            replacement[name] = float(predictive_pass_probability(m, variance, protocol))
        m = mean.copy()
        m[1:] = truth[1:]
        replacement["both_true_means_keep_epistemic_variance"] = float(predictive_pass_probability(m, variance, protocol))
        psi, reference, full, structural, bank_truth, target_seed = restore_context(record, protocol)
        eta = TransferMixtureGPR([structural[2], full[2]], psi, reference)
        visits = np.zeros(len(psi), dtype=int)
        batches = []
        for step, candidate in enumerate(record["row"]["queries"]):
            x = [[candidate]]
            mu = float(eta.posterior_mean_many(x)[0])
            v = float(eta.posterior_var_many(x)[0])
            forecast = float(np.exp(mu + .5 * v) / protocol["replicates"])
            response, noise = observe(bank_truth[candidate], target_seed, candidate, int(visits[candidate]),
                                      protocol["replicates"])
            if candidate == scored["recommendation"]:
                batches.append({"step": step + 1, "g_error": float(response[1] - bank_truth[candidate, 1]),
                    "eta_error": float(response[2] - bank_truth[candidate, 2]),
                    "sample_g_noise": float(noise[1]), "predicted_g_noise": forecast,
                    "predicted_over_sample_g_noise": forecast / float(noise[1]),
                    "sample_variance_over_truth": float(noise[1] * protocol["replicates"] /
                                                       np.exp(bank_truth[candidate, 2]))})
            eta.update([candidate], response[2], noise[2])
            visits[candidate] += 1
        tau = np.asarray(record["source_info"][2]["source_function_variance"])
        names = structural[2].basis.names
        diagnostics.append({"seed": seed, "regime": regime, "update": update,
            "recommendation": scored["recommendation"], "predicted_power": scored["predicted_verifier_power"],
            "true_power": scored["true_verifier_power"], "standardized_error": scored["standardized_error"],
            "truth_mean_replacements": replacement, "selected_candidate_batches": batches,
            "source_risk_top_four": [names[i] for i in np.argsort(-tau)[:4]],
            "target_risk_support": [names[i] for i in simulator_supports(seed, regime, protocol["psi_dim"])[2]]})
    result = {"scope": "Post-decision attribution only; no KG, new search or random verification. Truth replacements are not deployable rules.",
              "cases": diagnostics}
    (ARTIFACTS / "failure_attribution.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for r in diagnostics:
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
