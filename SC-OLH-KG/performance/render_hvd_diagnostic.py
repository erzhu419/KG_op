#!/usr/bin/env python3
"""Regenerate the secondary HVD table from its six compact aggregate rows."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render():
    with (ROOT / "paper_artifacts/hvd_diagnostic_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    domains = {"FactorShockStatePolicyRZDT1": "FactorShock", "InventorySupplyChain": "Inventory", "QueueResourceControl": "Queue"}
    lines = [r"\begin{tabular}{lllrrrr}", r"\toprule", r"Domain & Variance model & Feasible & Log-RMSE & Shape corr. & Mean verify \\", r"\midrule"]
    for domain, label in domains.items():
        for row in rows:
            if row["domain"] != domain:
                continue
            method = "Pooled" if row["method_identity"].endswith("hvd:pooled") else "Cumulative factor-HVD"
            lines.append(f"{label} & {method} & {row['true_feasible_count']}/{row['successful_rows']} & {float(row['mean_aleatoric_log_variance_rmse']):.3f} & {float(row['mean_aleatoric_variance_shape_correlation']):.3f} & {float(row['mean_target_verification_calls']):.1f} \\\\")
        if label != "Queue":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    destination = ROOT / "manuscript/tables/hvd_diagnostic.tex"
    source = "\n".join(lines)
    changed = destination.read_text() != source
    destination.write_text(source)
    print(f"HVD table regenerated from {len(rows)} aggregate rows; changed={changed}")


if __name__ == "__main__":
    render()
