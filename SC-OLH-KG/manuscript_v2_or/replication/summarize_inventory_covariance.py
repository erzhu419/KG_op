"""Paired inventory policy-ranking contrasts from the archived eight-observation diagnostic."""

import argparse
import json
from pathlib import Path

import numpy as np

from summarize_metaprior_opsd_panel import _paired_stats


def summarize(audit_dir: Path) -> dict:
    jobs = json.loads((audit_dir / "jobs.json").read_bytes())["jobs"]
    values = []
    for job in jobs:
        if job["family"] != "Inventory_CrossHorizon":
            continue
        item = json.loads((audit_dir / "items" / (job["job_id"] + ".json")).read_bytes())
        if item["job_id"] != job["job_id"] or item["target_task_idx"] != job["target_task_idx"]:
            raise ValueError(f"Inventory task receipt disagrees with its job: {job['job_id']}")
        values.append(item["models"])
    full = np.array([v["MetaPrior-FullSURE"]["spearman_obj"] for v in values])
    contrasts = {}
    for comparator in ("MetaPrior-DiagSURE", "TargetGP-RawPsi", "MetaPrior-Scrambled"):
        baseline = np.array([v[comparator]["spearman_obj"] for v in values])
        contrasts[comparator] = {
            "comparator_mean": float(np.mean(baseline)),
            **_paired_stats(full - baseline),
        }
    return {
        "family": "Inventory_CrossHorizon",
        "endpoint": "Objective Spearman correlation across the same held-out policies after eight observations",
        "analysis": "Exploratory task-paired FullSURE minus comparator; marginal 95% paired-t intervals",
        "full_mean": float(np.mean(full)),
        "contrasts": contrasts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.audit_dir), indent=2) + "\n", encoding="utf-8")
