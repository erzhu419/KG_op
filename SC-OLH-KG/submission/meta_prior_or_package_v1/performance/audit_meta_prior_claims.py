"""Audit manuscript claim boundaries against frozen meta-prior artifacts."""

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def _read(relative):
    return json.loads((ROOT / relative).read_text())


def run():
    manifest = _read("performance/meta_prior_submission_manifest.json")
    ledger = _read("performance/meta_prior_claims.json")
    manuscript = "\n".join((ROOT / "manuscript/meta_prior_main.tex").read_text().splitlines())
    section_text = "\n".join(path.read_text() for path in (ROOT / "manuscript/meta_prior_sections").glob("*.tex"))
    all_text = manuscript + "\n" + section_text
    errors = []
    required_files = [manifest["entrypoint"], manifest["supplement"], manifest["table_generator"], manifest["baseline_table_generator"], manifest["uncertainty_generator"], manifest["rbf_baseline_runner"], manifest["claim_audit"], manifest["claim_audit_report"]]
    required_files += list(manifest["source_of_truth"].values())
    for relative in required_files:
        if not (ROOT / relative).exists():
            errors.append(f"missing file: {relative}")

    development = _read("paper_artifacts/meta_prior_feasible_deployment_v1/source_structure_arm.json")
    if development.get("confirmation_block_read") is not False:
        errors.append("development source-structure artifact reads a confirmation block")
    if development.get("decision_performance_established") is not False:
        errors.append("development source-structure artifact promotes decision performance")
    confirmation = _read("paper_artifacts/meta_prior_feasible_deployment_v1/source_structure_confirmation_v2.json")
    if confirmation.get("confirmation_block_read") is not False:
        errors.append("independent v2 artifact reads the original confirmation block")
    if confirmation.get("decision_performance_established") is not False:
        errors.append("independent v2 artifact promotes universal decision performance")
    if confirmation.get("contract", {}).get("target_law_seeds") != [95800, 95801, 95802, 95803, 95804, 95805, 95806, 95807]:
        errors.append("independent v2 seed block changed")
    if len({row["law_seed"] for row in confirmation["rows"]}) != 8:
        errors.append("independent v2 rows do not retain all eight seeds")
    baseline = _read("paper_artifacts/meta_prior_feasible_deployment_v1/target_only_baseline_comparison.json")
    expected_blocks = {"development", "independent_v2"}
    if set(baseline.get("blocks", {})) != expected_blocks:
        errors.append("target-only baseline comparison is incomplete")
    for block_name, block in baseline.get("blocks", {}).items():
        for arm in ("source_selected_structure", "same_source_full_dictionary", "target_only_full_dictionary", "target_only_rbf_kg"):
            if block.get("summary", {}).get(arm, {}).get("tasks") != 8:
                errors.append(f"{block_name}: target-only comparison does not retain all eight tasks")
    if not (ROOT / "paper_artifacts/meta_prior_feasible_deployment_v1/paired_uncertainty.json").exists():
        errors.append("paired uncertainty artifact is missing")
    rbf = _read("paper_artifacts/meta_prior_feasible_deployment_v1/target_only_rbf_development.json")
    rbf_v2 = _read("paper_artifacts/meta_prior_feasible_deployment_v1/target_only_rbf_confirmation_v2.json")
    for name, artifact, seeds in (
        ("development RBF", rbf, [95425, 95426, 95427, 95428, 95430, 95431, 95432, 95434]),
        ("v2 RBF", rbf_v2, [95800, 95801, 95802, 95803, 95804, 95805, 95806, 95807]),
    ):
        if artifact.get("confirmation_block_read") is not False or artifact.get("decision_performance_established") is not False:
            errors.append(f"{name} artifact crosses its baseline boundary")
        if sorted(row["law_seed"] for row in artifact.get("rows", [])) != sorted(seeds):
            errors.append(f"{name} does not retain all eight seeds")
    risk = _read("paper_artifacts/meta_prior_risk_identifiability_v1/closed_loop.json")
    if any(row.get("confirmation_block_read") is not False for row in [risk]):
        errors.append("risk artifact confirmation boundary is not false")
    original_block = manifest["confirmation_boundary"]["original_block"]
    if manifest["confirmation_boundary"]["original_block_may_be_read"] is not False:
        errors.append("original confirmation block is not frozen")
    overlap = manifest["confirmation_boundary"].get("development_overlap", [])
    if overlap != [95434]:
        errors.append("original/development seed overlap is not recorded as 95434")

    forbidden_positive = [
        r"we (?:demonstrate|show|establish) (?:a )?learned trajectory",
        r"we (?:demonstrate|show|establish).*factor-level risk recovery",
        r"universal (?:deployment|optimization) advantage",
        r"arbitrary high-dimensional free-variable optimization",
    ]
    for pattern in forbidden_positive:
        if re.search(pattern, all_text, flags=re.I):
            errors.append(f"unsupported positive wording: {pattern}")
    required_phrases = [
        "declared public representation",
        "same-source full-dictionary",
        "not a stable independent-deployment claim",
        "factor-level risk recovery",
        "original confirmation interval",
        "was not run as a block",
        "95434 overlaps the development artifact",
        "independent fixed-seed confirmation",
        "target-only full-dictionary",
        "target-only rbf",
        "generic target-only rbf",
        "data and code accessibility",
        "paired bootstrap",
    ]
    normalized_text = re.sub(r"\s+", " ", all_text).lower()
    for phrase in required_phrases:
        if phrase.lower() not in normalized_text:
            errors.append(f"missing boundary phrase: {phrase}")

    statuses = {claim["id"]: claim["status"] for claim in ledger["claims"]}
    if statuses.get("C5") != "unsupported_and_excluded":
        errors.append("unsupported claim ledger entry C5 is not excluded")
    legacy_lines = [line.lower() for line in all_text.splitlines() if "atlas" in line.lower() or "energy" in line.lower()]
    if legacy_lines and any(
        all(token not in line for token in ("separate", "different", "own artifact identity"))
        for line in legacy_lines
    ):
        errors.append("legacy atlas/Energy evidence is not explicitly separated")
    if "supported_development_bounded_not_confirmed" not in statuses.values():
        errors.append("C3 is not marked as bounded and unconfirmed")
    return {"passed": not errors, "errors": errors, "claim_status": statuses,
            "original_confirmation_block_read": False,
            "independent_v2_block_read": True,
            "original_confirmation_block_run_as_block": False,
            "development_overlap": overlap,
            "overlap_eligible_as_clean_confirmation": False,
            "legacy_entrypoint_separate": "sections/04_atlas" not in manuscript,
            "legacy_evidence_lines": legacy_lines}


if __name__ == "__main__":
    result = run()
    output = ROOT / "paper_artifacts/meta_prior_manuscript_claim_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
