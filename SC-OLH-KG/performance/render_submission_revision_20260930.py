#!/usr/bin/env python3
"""Render current tables from completed original and September 30 evidence."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.render_or_review_final_artifacts import render as render_original, DEFAULT_EVIDENCE, DEFAULT_REVISION, DEFAULT_MANUSCRIPT, _write_energy_table
from performance.render_hvd_diagnostic import render as render_hvd

REVISION = ROOT / "paper_artifacts/submission_revision_20260930"
ENERGY = ROOT / "paper_artifacts/submission_revision_energy_v5_20260930"
LABELS = {"source_atlas": "Source-scored atlas", "standardized_generic_dct_maximin": "Medoid + structural", "medoid_augmented": "Medoid + augmented", "source_best_z": "Source-best + structural", "source_best_raw_l2": "Source-best + raw L2", "source_greedy_portfolio": "Source-greedy portfolio"}


def table(name, headings, rows):
    source = ["\\begin{tabular}{"+"l"+"r"*(len(headings)-1)+"}", r"\toprule", " & ".join(headings)+r" \\", r"\midrule"]
    source += [" & ".join(map(str,row))+r" \\" for row in rows]
    source += [r"\bottomrule", r"\end{tabular}", ""]
    (DEFAULT_MANUSCRIPT / "tables" / name).write_text("\n".join(source))


def main():
    profile = json.loads((REVISION / "profile_analysis.json").read_text())
    energy = json.loads((ENERGY / "energy_analysis.json").read_text())
    if profile["status"] != "complete" or energy["status"] != "complete":
        raise ValueError("complete statistical analyses required")
    manifest_path = DEFAULT_MANUSCRIPT / "review_artifact_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = render_original(DEFAULT_EVIDENCE, DEFAULT_MANUSCRIPT, skip_figures=True, revision=DEFAULT_REVISION)
    manifest["outputs"] += [r for r in previous.get("outputs",[]) if r["path"].startswith("figures/")]
    outputs = []
    confirm = next(s for s in profile["studies"] if s["study"] == "confirmation")
    rows = [[LABELS[r["arm"]], f"{r['true_feasible_coverage_count']}/240", f"{r['certified_true_feasible_deployment_count']}/240", r["false_certificate_count"], f"{r['mean_penalized_loss']:.3f}"] for r in confirm["summaries"]]
    table("review_confirmation.tex", ["Design", "Coverage", "True certificates", "False", "Loss"], rows)
    outputs.append("review_confirmation.tex")
    rows = []
    for study in profile["studies"]:
        if study["study"] != "attribution":
            continue
        for r in study["summaries"]:
            rows.append([f"{study['dimension']}: {LABELS[r['arm']]}", f"{r['true_feasible_coverage_count']}/160", f"{r['certified_true_feasible_deployment_count']}/160", f"{r['mean_penalized_loss']:.3f}"])
    table("review_attribution.tex", ["Resolution: design", "Coverage", "True certificates", "Loss"], rows)
    outputs.append("review_attribution.tex")
    rows = []
    for family in confirm["by_family"]:
        for r in family["summaries"]:
            if r["arm"] in ("source_atlas", "standardized_generic_dct_maximin", "source_greedy_portfolio"):
                rows.append([f"{family['family_seed']}: {LABELS[r['arm']]}", f"{r['true_feasible_coverage_count']}/80", f"{r['certified_true_feasible_deployment_count']}/80", f"{r['mean_penalized_loss']:.3f}"])
    table("review_confirmation_families.tex", ["Family: design", "Coverage", "True certificates", "Loss"], rows)
    outputs.append("review_confirmation_families.tex")
    rows = []
    for regime in confirm["by_regime"]:
        for r in regime["summaries"]:
            if r["arm"] in ("source_atlas", "standardized_generic_dct_maximin", "source_greedy_portfolio"):
                rows.append([regime["regime"].replace("_", " ")+": "+LABELS[r["arm"]], f"{r['true_feasible_coverage_count']}/30", f"{r['certified_true_feasible_deployment_count']}/30"])
    table("review_confirmation_regimes.tex", ["Mechanism: design", "Coverage", "True certificates"], rows)
    outputs.append("review_confirmation_regimes.tex")
    rows = []
    for study in profile["studies"]:
        if study["study"] == "library_size":
            for r in study["summaries"]:
                rows.append([f"{study['library_size']}: {LABELS[r['arm']]}", f"{r['true_feasible_coverage_count']}/80", f"{r['certified_true_feasible_deployment_count']}/80", f"{r['mean_penalized_loss']:.3f}"])
    table("review_library_sizes.tex", ["Library size: design", "Coverage", "True certificates", "Loss"], rows)
    outputs.append("review_library_sizes.tex")
    rows = [[LABELS[r["arm"]], r["source_calls_once"], r["cumulative_calls"], f"{r['calls_per_true_certificate']:.1f}"] for r in confirm["actual_archive_reuse_cost"]]
    table("review_confirmation_cost.tex", ["Design", "Source calls", "Cumulative calls", "Calls/true certificate"], rows)
    outputs.append("review_confirmation_cost.tex")
    rows = [[r["arm"].replace("_", " "), r["certified_count"], r["block_stable_count"], r["nonoverlap_stable_count"], r["joint_stable_count"]] for r in energy["temporal_summaries"]]
    table("review_temporal_v5.tex", ["Design", "Certified", "Block", "Nonoverlap", "Joint"], rows)
    outputs.append("review_temporal_v5.tex")
    rows = []
    for arm in ("source_atlas", "generic_dct_maximin", "standardized_generic_dct_maximin", "source_greedy_portfolio"):
        selected = [r for r in energy["actual_archive_reuse"] if r["arm"] == arm]
        calls = sum(r["cumulative_calls"] for r in selected)
        certificates = sum(r["expected_certificates"] for r in selected)
        rows.append([arm.replace("_"," "), sum(r["source_calls_once"] for r in selected), f"{calls:.1f}", f"{certificates:.1f}", f"{calls/certificates:.1f}"])
    table("review_energy_reuse.tex", ["Design", "Source calls", "Cumulative calls", "Expected certificates", "Calls/certificate"], rows)
    outputs.append("review_energy_reuse.tex")
    _write_energy_table(energy, DEFAULT_MANUSCRIPT / "tables/review_energy_v3.tex")
    outputs.append("review_energy_v3.tex")
    render_hvd()
    outputs.append("hvd_diagnostic.tex")
    generated = {"tables/"+name for name in outputs}
    manifest["outputs"] = [r for r in manifest["outputs"] if r["path"] not in generated] + [{"path":name} for name in sorted(generated)]
    manifest["contracts"]["output_hashes_cover_all_generated_tables_and_figures"] = False
    manifest["revision_inputs"] = ["paper_artifacts/submission_revision_20260930/profile_analysis.json", "paper_artifacts/submission_revision_energy_v5_20260930/energy_analysis.json", "paper_artifacts/hvd_diagnostic_summary.csv"]
    manifest["current_external_contract"] = energy["contract_id"]
    manifest["prospective_confirmation_task_count"] = 240
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    print("Current revision tables generated:",len(outputs))


if __name__ == "__main__":
    main()
