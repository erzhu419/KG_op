"""Create paired bootstrap intervals for every metric reported in the manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 100_000


def _bootstrap(values, rng):
    values = np.asarray(values, dtype=float)
    draws = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci95_percentile": [float(x) for x in np.percentile(means, [2.5, 97.5])],
        "paired_values": [float(x) for x in values],
    }


def _read(path):
    return json.loads((ROOT / path).read_text())


def run(output_artifact=None, output_table=None):
    rng = np.random.default_rng(20261004)
    rep = _read("paper_artifacts/meta_prior_feasible_deployment_v1/representation_initial.json")
    policy = {r["law_seed"]: r for r in rep["arms"]["policy"]}
    observable = {r["law_seed"]: r for r in rep["arms"]["observable"]}
    metrics = {}
    for key, label in (
        ("f_rmse_holdout", "observable minus policy objective RMSE"),
        ("g_rmse_holdout", "observable minus policy constraint RMSE"),
        ("eta_rmse_holdout", "observable minus policy log-risk RMSE"),
        ("g_rank_spearman_holdout", "observable minus policy constraint rank correlation"),
    ):
        values = [observable[s][key] - policy[s][key] for s in sorted(policy)]
        metrics[label] = _bootstrap(values, rng)

    calib = _read("paper_artifacts/meta_prior_risk_identifiability_v1/initial_calibration.json")
    for key, label in (("eta_rmse_holdout", "learned minus pooled log-risk RMSE"), ("factor_beta_rmse", "learned minus pooled factor coefficient RMSE")):
        values = [r["learned"][key] - r["pooled"][key] for r in calib["target_rows"]]
        metrics[label] = _bootstrap(values, rng)

    comparison = _read("paper_artifacts/meta_prior_feasible_deployment_v1/target_only_baseline_comparison.json")
    for block_name, block in comparison["blocks"].items():
        for name, entry in block["comparisons"].items():
            metrics[f"{block_name}: {name}"] = _bootstrap(entry["deltas"], rng)

    result = {
        "artifact_id": "meta_prior_paired_bootstrap_v1",
        "method": "seed-level paired percentile bootstrap; 100000 resamples; RNG seed 20261004",
        "interpretation": "descriptive intervals for eight-task blocks; not a universal-superiority test",
        "metrics": metrics,
    }
    artifact_path = ROOT / (output_artifact or "paper_artifacts/meta_prior_feasible_deployment_v1/paired_uncertainty.json")
    artifact_path.write_text(json.dumps(result, indent=2) + "\n")
    rows = []
    for label, stat in metrics.items():
        safe_label = label.replace("_", r"\_")
        rows.append("%s & %d & %+.5f & [%+.5f, %+.5f] \\\\\n" % (safe_label, stat["n"], stat["mean"], *stat["ci95_percentile"]))
    table = r"""\begin{tabular}{p{0.56\textwidth}r r l}
\toprule
Metric & $n$ & Mean difference & Percentile bootstrap 95\% interval\\
\midrule
""" + "".join(rows) + r"""\bottomrule
\end{tabular}
"""
    table_path = ROOT / (output_table or "manuscript/meta_prior_uncertainty_table.tex")
    table_path.write_text(table)
    return {"artifact": str(artifact_path), "table": str(table_path), "metric_count": len(metrics)}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
