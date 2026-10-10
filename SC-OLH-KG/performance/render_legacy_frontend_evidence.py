"""Reuse compact frozen summaries for the legacy frontend component evidence."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render():
    summary_path = ROOT / "paper_artifacts/paper_result_audit_summary_v1.csv"
    with summary_path.open() as file:
        summaries = list(csv.DictReader(file))
    arms = [
        ("Combined legacy frontend", "Proposal only", "final_frozen_source_frontend_backend_d1000_n13", "frozen_crossdim_proposal_only"),
        ("Combined legacy frontend", "Stacked GP", "final_frozen_source_frontend_backend_d1000_n13", "stacked_transfer_gp_cbo:official_transfergpbo_code"),
        ("Combined legacy frontend", "Canonical SAASBO", "final_frozen_source_frontend_backend_d1000_n13", "canonical_saasbo_every_iteration"),
        ("Common Sobol", "Proposal only", "final_frozen_sobol_frontend_control_d1000_n13", "common_sobol_proposal_only"),
        ("Common Sobol", "Stacked GP", "final_frozen_sobol_frontend_control_d1000_n13", "stacked_transfer_gp_cbo:official_transfergpbo_code"),
        ("Common Sobol", "Canonical SAASBO", "final_frozen_sobol_frontend_control_d1000_n13", "canonical_saasbo_every_iteration"),
    ]
    rows = []
    for frontend, backend, track, method in arms:
        matches = [r for r in summaries if r["track_id"] == track and r["method_identity"] == method]
        rows.append({"frontend": frontend, "backend": backend, "track_id": track, "method_identity": method,
            "runs": sum(int(r["submitted_rows"]) for r in matches),
            "true_feasible": sum(int(r["true_feasible_count"]) for r in matches),
            "certified": sum(int(r["certified_count"]) for r in matches),
            "false_certificates": sum(int(r["false_certificate_count"]) for r in matches),
            "source_calls_per_archive": float(matches[0]["mean_source_calls"]),
            "target_search_calls_per_run": float(matches[0]["mean_target_search_calls"]),
            "mean_verification_calls": sum(float(r["mean_target_verification_calls"]) * int(r["submitted_rows"]) for r in matches)
                / sum(int(r["submitted_rows"]) for r in matches),
            "per_domain": {r["domain"]: {"runs": int(r["submitted_rows"]), "certified": int(r["certified_count"])} for r in matches}})
    native_path = ROOT / "paper_artifacts/or_review/native_transfer.json"
    native = json.loads(native_path.read_text())["aggregate_analysis"]
    payload = {"study_role": "retained legacy frontend component study; no new experiment",
        "source_summary": str(summary_path.relative_to(ROOT)), "native_summary": str(native_path.relative_to(ROOT)),
        "domains": ["FactorShockStatePolicyRZDT1", "InventorySupplyChain", "QueueResourceControl"],
        "target_dimension": 1000, "algorithmic_seeds_per_fixed_domain": 20,
        "frontend_backend_rows": rows, "native_transfer_rows": native["method_summaries"],
        "native_best_certified": max(r["certified_safe_count"] for r in native["method_summaries"]),
        "scope": "historical v69 Gaussian terminal contract; seed repeatability within three fixed task families; separate from physical Inventory and current meta-prior claims",
        "current_physical_inventory_uses_legacy_frontend": False}
    components = [r for r in summaries if r["track_id"] == "frontend_component_causal_ablation_d1000_n10"]
    payload["historical_frontend_implementation"] = "LearnedMetaPrior.risk_objective_initial_candidates; distinct from SourceScoredProfileAtlas/profile_atlas_v2"
    payload["frontend_component_rows"] = [{"method_identity": r["method_identity"], "domain": r["domain"],
        "runs": int(r["submitted_rows"]), "certified": int(r["certified_count"])} for r in components]
    (ROOT / "paper_artifacts/legacy_frontend_evidence.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Frontend & Backend & Source & Search & Certified & False \\", r"\midrule"]
    for row in rows:
        lines.append(f"{row['frontend']} & {row['backend']} & {int(row['source_calls_per_archive'])} & "
            f"{int(row['target_search_calls_per_run'])} & {row['certified']}/{row['runs']} & {row['false_certificates']} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "manuscript/meta_prior_legacy_frontend_table.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps({"frontend_backend_rows": rows, "native_best_certified": payload["native_best_certified"]}, indent=2))
    return payload


if __name__ == "__main__":
    render()
