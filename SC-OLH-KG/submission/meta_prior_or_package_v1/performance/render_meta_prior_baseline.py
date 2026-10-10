"""Render the same-contract target-only baseline comparison.

This reads only frozen JSON artifacts.  It checks that the target-only arm
uses exactly the seed block and observation contract of the paired source
arms, then writes a compact machine-readable comparison and manuscript table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "paper_artifacts/meta_prior_feasible_deployment_v1"


def _read(name: str):
    return json.loads((ARTIFACT_DIR / name).read_text())


def _block(label, source_file, target_file, rbf_file, expected_seeds):
    source = _read(source_file)
    target = _read(target_file)
    rbf = _read(rbf_file)
    for artifact in (source, target, rbf):
        if artifact.get("confirmation_block_read") is True or artifact.get("original_confirmation_block_read") is True:
            raise ValueError(f"{label}: original confirmation block was read")
    source_rows = source["rows"]
    target_rows = target["rows"]
    target_rows = {row["law_seed"]: row for row in target_rows}
    rbf_rows = {row["law_seed"]: row for row in rbf["rows"]}
    seeds = sorted(expected_seeds)
    for arm in ("source_selected_structure", "same_source_full_dictionary"):
        rows = [row for row in source_rows if row["arm"] == arm]
        if sorted(row["law_seed"] for row in rows) != seeds:
            raise ValueError(f"{label}: {arm} seed block changed")
    if sorted(target_rows) != seeds:
        raise ValueError(f"{label}: target-only seed block changed")
    if sorted(rbf_rows) != seeds:
        raise ValueError(f"{label}: RBF seed block changed")
    rows = {}
    for seed in seeds:
        rows[seed] = {
            "source_selected_structure": next(r for r in source_rows if r["arm"] == "source_selected_structure" and r["law_seed"] == seed),
            "same_source_full_dictionary": next(r for r in source_rows if r["arm"] == "same_source_full_dictionary" and r["law_seed"] == seed),
            "target_only_full_dictionary": target_rows[seed],
            "target_only_rbf_kg": rbf_rows[seed],
        }
    summary = {}
    for arm in ("source_selected_structure", "same_source_full_dictionary", "target_only_full_dictionary", "target_only_rbf_kg"):
        data = [rows[s][arm] for s in seeds]
        summary[arm] = {
            "tasks": len(data),
            "mean_final_true_reward": sum(r["final_true_expected_reward"] for r in data) / len(data),
            "verified": sum(bool(r["verified"]) for r in data),
            "false_certificates": sum(bool(r.get("false_certificate", False)) for r in data),
        }
    comparisons = {}
    for left, right in (
        ("source_selected_structure", "same_source_full_dictionary"),
        ("source_selected_structure", "target_only_full_dictionary"),
        ("same_source_full_dictionary", "target_only_full_dictionary"),
        ("source_selected_structure", "target_only_rbf_kg"),
        ("same_source_full_dictionary", "target_only_rbf_kg"),
        ("target_only_full_dictionary", "target_only_rbf_kg"),
    ):
        key = f"{left}_minus_{right}"
        deltas = [rows[s][left]["final_true_expected_reward"] - rows[s][right]["final_true_expected_reward"] for s in seeds]
        comparisons[key] = {
            "deltas": deltas,
            "mean": sum(deltas) / len(deltas),
            "positive": sum(d > 0 for d in deltas),
            "negative": sum(d < 0 for d in deltas),
            "ties": sum(d == 0 for d in deltas),
        }
    return {"seed_block": seeds, "summary": summary, "comparisons": comparisons}


def render(output_artifact=None, output_table=None):
    development = _block(
        "development",
        "source_structure_arm.json",
        "target_only_development.json",
        "target_only_rbf_development.json",
        [95425, 95426, 95427, 95428, 95430, 95431, 95432, 95434],
    )
    independent = _block(
        "independent_v2",
        "source_structure_confirmation_v2.json",
        "target_only_confirmation_v2.json",
        "target_only_rbf_confirmation_v2.json",
        [95800, 95801, 95802, 95803, 95804, 95805, 95806, 95807],
    )
    result = {
        "artifact_id": "meta_prior_same_contract_baseline_comparison_v2",
        "source": {
            "development_source": "paper_artifacts/meta_prior_feasible_deployment_v1/source_structure_arm.json",
            "development_target_only": "paper_artifacts/meta_prior_feasible_deployment_v1/target_only_development.json",
            "independent_source": "paper_artifacts/meta_prior_feasible_deployment_v1/source_structure_confirmation_v2.json",
            "independent_target_only": "paper_artifacts/meta_prior_feasible_deployment_v1/target_only_confirmation_v2.json",
            "development_rbf": "paper_artifacts/meta_prior_feasible_deployment_v1/target_only_rbf_development.json",
            "independent_rbf": "paper_artifacts/meta_prior_feasible_deployment_v1/target_only_rbf_confirmation_v2.json",
        },
        "blocks": {"development": development, "independent_v2": independent},
        "interpretation": "same-contract baseline comparison; no deployment promotion",
    }
    artifact_path = ROOT / (output_artifact or "paper_artifacts/meta_prior_feasible_deployment_v1/target_only_baseline_comparison.json")
    artifact_path.write_text(json.dumps(result, indent=2) + "\n")

    rows = []
    for label, block in (("Development", development), ("Independent v2", independent)):
        s = block["summary"]
        rows.append(
            "%s & %.5f (%d/8) & %.5f (%d/8) & %.5f (%d/8) & %.5f (%d/8) & %+.5f \\\\\n" % (
                label,
                s["source_selected_structure"]["mean_final_true_reward"], s["source_selected_structure"]["verified"],
                s["same_source_full_dictionary"]["mean_final_true_reward"], s["same_source_full_dictionary"]["verified"],
                s["target_only_full_dictionary"]["mean_final_true_reward"], s["target_only_full_dictionary"]["verified"],
                s["target_only_rbf_kg"]["mean_final_true_reward"], s["target_only_rbf_kg"]["verified"],
                block["comparisons"]["source_selected_structure_minus_target_only_rbf_kg"]["mean"],
            )
        )
    table = r"""\begin{tabular}{lrrrrr}
\toprule
Block & Source-selected & Full dictionary & Target-only & RBF-KG & Source--RBF\\
 & mean (certified) & mean (certified) & mean (certified) & mean (certified) & mean difference\\
\midrule
""" + "".join(rows) + r"""\bottomrule
\end{tabular}
"""
    table_path = ROOT / (output_table or "manuscript/meta_prior_baseline_table.tex")
    table_path.write_text(table)
    return {"artifact": str(artifact_path), "table": str(table_path), "blocks": result["blocks"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--table", default=None)
    args = parser.parse_args()
    print(json.dumps(render(args.artifact, args.table), indent=2))
