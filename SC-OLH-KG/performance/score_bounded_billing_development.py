"""Analytic scoring after bounded-price inference and numerical refinement."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.verification_kg import verification_utility
from problems.inventory_cost_plan import InventoryCostSimulator, InventoryPlan, sample_cost_law


def score(directory=ROOT / "paper_artifacts/bounded_billing_replay_v1"):
    directory = Path(directory)
    replay = json.loads((directory / "replay.json").read_text())
    if not replay["numerical_gate_passed"]:
        raise ValueError("Refine numerical integration before scoring target truth")
    p = replay["contract"]
    archive = json.loads((ROOT / p["source_archive"]).read_text())
    policies = np.asarray(archive["public_policies"])
    prior_score = json.loads((ROOT / "paper_artifacts/billing_offset_replay_v1/development_scoring.json").read_text())
    rows, physical, reproduction_error = [], [], 0.
    for seed in sorted({r["law_seed"] for r in replay["rows"]}):
        world = InventoryCostSimulator(InventoryPlan(**archive["contexts"][0]), sample_cost_law(seed, p["constraint_price_bounds"]))
        for row in replay["rows"]:
            if row["law_seed"] != seed:
                continue
            for stage in row["stages"]:
                for mode, candidate in [("mixed", stage)] + ([("physical", stage["physical_only"])] if row["arm"] in p["arms"] else []):
                    truth = world.oracle_moments(policies[candidate["recommendation"]])
                    utility, power = verification_utility(truth[None], np.zeros((1, 3)), p)
                    result = {"law_seed": seed, "arm": row["arm"], "paid_points": stage["paid_points"],
                        "recommendation": candidate["recommendation"], "true_reward": float(utility[0]),
                        "true_verifier_power": float(power[0]), "predicted_verifier_power": candidate["predicted_verifier_power"],
                        "absolute_power_error": abs(candidate["predicted_verifier_power"] - float(power[0]))}
                    if "recommendation_g_eta_mean" in candidate:
                        mean = np.asarray(candidate["recommendation_g_eta_mean"])
                        covariance = np.asarray(candidate["recommendation_g_eta_covariance"])
                        result.update(truth_g_eta=truth[1:].tolist(), posterior_g_eta_mean=mean.tolist(),
                            mean_error_in_posterior_sd=((mean - truth[1:]) / np.sqrt(np.diag(covariance))).tolist())
                    if mode == "physical":
                        physical.append(result)
                    else:
                        rows.append(result)
                        if row["arm"] not in p["arms"]:
                            old = next(r for r in prior_score["rows"] if r["law_seed"] == seed and r["arm"] == row["arm"]
                                and r["paid_points"] == stage["paid_points"])
                            reproduction_error = max(reproduction_error, abs(result["true_reward"] - old["true_reward"]))
    if reproduction_error > 1e-10:
        raise ValueError("Bounded scorer changed the previous development task contract")
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    summary = [{"arm": arm, "paid_points": n, **{key: float(np.mean([r[value] for r in rows if r["arm"] == arm and r["paid_points"] == n]))
        for key, value in (("mean_true_reward", "true_reward"), ("mean_absolute_power_error", "absolute_power_error"))}}
        for arm in arms for n in p["stages"]]
    seeds = sorted({r["law_seed"] for r in rows})
    resamples = np.random.default_rng(1005).integers(len(seeds), size=(5000, len(seeds)))
    comparisons = [("bounded_source", control) for control in arms if control != "bounded_source"]
    comparisons.append(("bounded_uniform_prices", "bounded_uniform_pooled_risk"))
    paired = []
    for selected, control in comparisons:
        for n in p["stages"]:
            left = {r["law_seed"]: r["true_reward"] for r in rows if r["arm"] == selected and r["paid_points"] == n}
            right = {r["law_seed"]: r["true_reward"] for r in rows if r["arm"] == control and r["paid_points"] == n}
            difference = np.array([left[s] - right[s] for s in seeds])
            paired.append({"selected": selected, "control": control, "paid_points": n,
                "paired_mean_reward_difference": float(difference.mean()),
                "development_bootstrap_95_interval": np.percentile(difference[resamples].mean(axis=1), [2.5, 97.5]).tolist(),
                "wins_ties_losses": [int(np.sum(difference > 1e-10)), int(np.sum(np.abs(difference) <= 1e-10)), int(np.sum(difference < -1e-10))],
                "per_seed_difference": {str(s): float(d) for s, d in zip(seeds, difference)}})
    kg_pairs = []
    for seed in seeds:
        task = {r["arm"]: r for r in replay["rows"] if r["law_seed"] == seed}
        for control in p["arms"][1:]:
            for left, right in zip(task["bounded_source"]["stages"], task[control]["stages"]):
                kg_pairs.append({"law_seed": seed, "control": control, "paid_points": left["paid_points"],
                    "source_probe": left["next_kg_probe"], "control_probe": right["next_kg_probe"],
                    "probe_changed": left["next_kg_probe"] != right["next_kg_probe"]})
    result = {"study_role": "post-inference analytic scoring of opened development laws",
        "target_truth_used_for_scoring_only": True, "independent_confirmation": False,
        "new_source_calls": 0, "new_target_calls": 0, "new_verification_calls": 0,
        "baseline_reproduction_error": reproduction_error, "summary": summary, "paired": paired,
        "rows": rows, "physical_only_rows": physical, "unexecuted_kg_pairs": kg_pairs}
    (directory / "development_scoring.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": summary, "paired": paired}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=ROOT / "paper_artifacts/bounded_billing_replay_v1")
    score(parser.parse_args().directory)
