"""Source-task holdout prediction for compatible risk transfer diagnosis."""
import numpy as np

from performance.audit_meta_prior_risk_family_prequential import model_log_scores
from performance.risk_family_reweighting import family_predictive_branches
from performance.target_risk_direction_mixture import joint_label_log_scores
from representation.source_function_prior import SourceFunctionPrior, head_batches
from representation.source_risk_ard import fit_source_risk_ard


def fit_source_fold(tasks, held_task, reference):
    training_ids = [i for i in range(len(tasks)) if i != held_task]
    training = [tasks[i] for i in training_ids]
    batches = head_batches(training, 2)
    noise = [t["noise"][:, 2] for t in training]
    full = SourceFunctionPrior.fit(batches, noise, reference, "fixed_source")
    structural = fit_source_risk_ard(batches, noise, reference)
    info = {"held_task": held_task, "training_tasks": training_ids,
        "source_label_count": sum(len(t["psi"]) for t in training),
        "optimizer_converged": structural.diagnostics["optimizer_converged"],
        "optimizer_iterations": structural.diagnostics["optimizer_iterations"]}
    return full, structural, info


def forecast_held_source(priors, task, families, nodes=32, steps=range(9, 25)):
    psi, labels, noise = task["psi"], task["response"][:, 2], task["noise"][:, 2]
    ids = np.arange(len(psi))
    forecasts = []
    for step in steps:
        t = step - 1
        weights, means, variances, summary = family_predictive_branches(priors, psi,
            ids[:t], labels[:t], noise[:t], t, families, nodes)
        # Held-function future labels reach only scoring, never the predictor.
        scores = joint_label_log_scores(means, variances, labels[t:t + 1], noise[t:t + 1])
        forecasts.append({"step": int(step), "candidate": int(t),
            **model_log_scores(weights, scores, len(priors), len(families), nodes),
            **summary})
    return forecasts


def summarize_source_holdout(rows, spec):
    cohort = spec["cohort"]
    seeds = cohort["seeds"]
    indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(len(seeds),
        size=(20000, len(seeds)))
    groups, contrasts = [], []
    for archive in cohort["source_archive_groups"]:
        source_by_arm = {}
        for arm in cohort["priors"]:
            archives = []
            for seed in seeds:
                folds = [r for r in rows if (r["seed"], r["archive_group"], r["arm"]) == (seed, archive, arm)]
                if sorted(r["held_task"] for r in folds) != list(range(cohort["held_tasks_per_archive"])):
                    raise ValueError("A11 summary requires every planned source task")
                archives.append({k: float(np.mean([r[k] for r in folds])) for k in
                    ("source_mean_log_score", "adapted_mean_log_score", "a9_mean_log_score")})
            source = np.array([a["source_mean_log_score"] for a in archives])
            source_by_arm[arm] = source
            for name in ("adapted", "a9"):
                gains = np.array([a[name + "_mean_log_score"] for a in archives]) - source
                groups.append({"archive_group": archive, "arm": arm, "contrast": name + "_minus_source",
                    "seed_archives": len(seeds), "held_tasks": len(seeds) * cohort["held_tasks_per_archive"],
                    "mean_gain": float(gains.mean()),
                    "paired_seed_bootstrap95": np.quantile(gains[indices].mean(axis=1), [.025, .975]).tolist(),
                    "better": int(np.sum(gains > 1e-12)), "worse": int(np.sum(gains < -1e-12)),
                    "seed_gains": gains.tolist()})
        gain = source_by_arm["risk_ard_mixture"] - source_by_arm["fixed_source"]
        contrasts.append({"archive_group": archive, "contrast": "learned_source_minus_generic_source",
            "scope": "Secondary diagnostic; both priors predict the same held source functions and points.",
            "mean_gain": float(gain.mean()),
            "paired_seed_bootstrap95": np.quantile(gain[indices].mean(axis=1), [.025, .975]).tolist(),
            "better": int(np.sum(gain > 1e-12)), "worse": int(np.sum(gain < -1e-12)),
            "seed_gains": gain.tolist()})
    return {"groups": groups, "source_contrasts": contrasts}
