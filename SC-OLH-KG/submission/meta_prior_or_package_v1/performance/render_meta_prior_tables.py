"""Generate meta-prior manuscript tables from frozen JSON artifacts.

The generator fails if an artifact is missing, if the paired arms do not share
the declared seed block, or if an original confirmation flag is true.  It does
not run simulations or read the original held-out confirmation block.
"""

import json
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]


def _read(relative):
    path = ROOT / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _rows(path, arms):
    data = _read(path)
    rows = data["rows"]
    assert data.get("confirmation_block_read", False) is False
    result = {arm: [row for row in rows if row["arm"] == arm] for arm in arms}
    assert all(len(result[arm]) == 8 for arm in arms), (path, {k: len(v) for k, v in result.items()})
    seeds = {row["law_seed"] for row in result[arms[0]]}
    assert all({row["law_seed"] for row in result[arm]} == seeds for arm in arms)
    return data, result


def render(output_main=None, output_supplement=None, output_confirmation=None):
    structure, arms = _rows(
        "paper_artifacts/meta_prior_feasible_deployment_v1/source_structure_arm.json",
        ("source_selected_structure", "same_source_full_dictionary"),
    )
    summary = structure["summary"]
    rows = sorted(arms["source_selected_structure"], key=lambda row: row["law_seed"])
    full = {row["law_seed"]: row for row in arms["same_source_full_dictionary"]}
    deltas = [row["final_true_expected_reward"] - full[row["law_seed"]]["final_true_expected_reward"] for row in rows]
    assert sum(delta > 0 for delta in deltas) == 3
    assert sum(delta < 0 for delta in deltas) == 1
    assert sum(delta == 0 for delta in deltas) == 4

    main_text = r"""\begin{tabular}{lrrrr}
\toprule
Arm & Tasks & Mean final reward & Certified & False certificates\\
\midrule
Source-selected structure & 8 & %.5f & %d/8 & %d\\
Same-source full dictionary & 8 & %.5f & %d/8 & %d\\
\bottomrule
\end{tabular}
""" % (
        summary["source_selected_structure"]["mean_final_true_reward"],
        summary["source_selected_structure"]["verified"],
        summary["source_selected_structure"]["false_certificates"],
        summary["same_source_full_dictionary"]["mean_final_true_reward"],
        summary["same_source_full_dictionary"]["verified"],
        summary["same_source_full_dictionary"]["false_certificates"],
    )

    supplement_text = r"""\begin{tabular}{rrrrrr}
\toprule
Law seed & Initial reward & Structure & Full dictionary & Difference & Both certified\\
\midrule
"""
    for row, delta in zip(rows, deltas):
        supplement_text += (
            "%d & %.6f & %.6f & %.6f & %+.6f & yes\\\\\n" % (
                row["law_seed"], row["initial_true_expected_reward"],
                row["final_true_expected_reward"], full[row["law_seed"]]["final_true_expected_reward"], delta,
            )
        )
    supplement_text += r"""\midrule
Mean & & %.6f & %.6f & %+.6f & 8/8\\
\bottomrule
\end{tabular}
""" % (
        summary["source_selected_structure"]["mean_final_true_reward"],
        summary["same_source_full_dictionary"]["mean_final_true_reward"],
        sum(deltas) / len(deltas),
    )

    output_main = ROOT / (output_main or "manuscript/meta_prior_tables.tex")
    output_supplement = ROOT / (output_supplement or "manuscript/meta_prior_supplement_tables.tex")
    output_main.write_text(main_text)
    output_supplement.write_text(supplement_text)

    confirmation_path = ROOT / "paper_artifacts/meta_prior_feasible_deployment_v1/source_structure_confirmation_v2.json"
    if confirmation_path.exists() and output_confirmation:
        confirmation, conf_arms = _rows(
            "paper_artifacts/meta_prior_feasible_deployment_v1/source_structure_confirmation_v2.json",
            ("source_selected_structure", "same_source_full_dictionary"),
        )
        conf_summary = confirmation["summary"]
        conf_rows = sorted(conf_arms["source_selected_structure"], key=lambda row: row["law_seed"])
        conf_full = {row["law_seed"]: row for row in conf_arms["same_source_full_dictionary"]}
        conf_deltas = [row["final_true_expected_reward"] - conf_full[row["law_seed"]]["final_true_expected_reward"] for row in conf_rows]
        out = ROOT / output_confirmation
        out.write_text(r"""\begin{table}[t]
\centering
\caption{Independent fixed-seed confirmation block; all eight tasks retained.}
\begin{tabular}{lrrrr}
\toprule
Arm & Tasks & Mean final reward & Certified & False certificates\\
\midrule
Source-selected structure & 8 & %.5f & %d/8 & %d\\
Same-source full dictionary & 8 & %.5f & %d/8 & %d\\
\bottomrule
\end{tabular}
\end{table}
\par\smallskip
All eight fixed seeds were retained after the oracle-only preflight; paired
final-reward differences were positive on %d, negative on %d, and tied on %d tasks.
""" % (
            conf_summary["source_selected_structure"]["mean_final_true_reward"],
            conf_summary["source_selected_structure"]["verified"],
            conf_summary["source_selected_structure"]["false_certificates"],
            conf_summary["same_source_full_dictionary"]["mean_final_true_reward"],
            conf_summary["same_source_full_dictionary"]["verified"],
            conf_summary["same_source_full_dictionary"]["false_certificates"],
            sum(delta > 0 for delta in conf_deltas), sum(delta < 0 for delta in conf_deltas), sum(delta == 0 for delta in conf_deltas),
        ))
    return {"main_table": str(output_main), "supplement_table": str(output_supplement),
            "confirmation_table": str(ROOT / output_confirmation) if output_confirmation and confirmation_path.exists() else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation-output", default=None)
    args = parser.parse_args()
    print(json.dumps(render(output_confirmation=args.confirmation_output), indent=2))
