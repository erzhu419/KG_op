import json
from pathlib import Path

from performance.analyze_or_review_v2_diagnostics import (
    build_aligned_geometry_audit,
    build_outcome_adjusted_cost_audit,
    build_task_seed_strata_audit,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "paper_artifacts/or_review"


def _load(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def test_task_seed_audit_distinguishes_latent_tasks_from_resolution_cells():
    audit = build_task_seed_strata_audit(task_count=20)
    assert sum(audit["overall_task_seed_mod_5_counts"]) == 160
    assert all(row["unique_seed_count"] == 20 for row in audit["rows"])
    assert audit["inference_contract"][
        "same_latent_task_seeds_are_crossed_with_three_resolutions"
    ] is True
    assert audit["inference_contract"][
        "pooled_480_cell_rates_are_descriptive"
    ] is True


def test_outcome_adjusted_cost_charges_archive_and_certification_probability():
    audit = build_outcome_adjusted_cost_audit(
        _load("randomized_profile_primary.json"),
        _load("randomized_profile_equal_preverification.json"),
    )
    rows = audit["matrices"]["equal_preverification"]["rows"]
    source = next(row for row in rows if row["arm"] == "source_atlas")
    sobol = next(row for row in rows if row["arm"] == "raw_sobol")
    assert source["source_archive_calls"] == 384
    assert sobol["source_archive_calls"] == 0
    assert source["amortization"]["20"][
        "expected_calls_per_certified_success"
    ] < sobol["amortization"]["20"][
        "expected_calls_per_certified_success"
    ]


def test_aligned_geometry_audit_is_postdecision_and_reproducible():
    first = build_aligned_geometry_audit(dimensions=(200,), task_count=2)
    second = build_aligned_geometry_audit(dimensions=(200,), task_count=2)
    assert first["design_fingerprint_root_sha256"] == second[
        "design_fingerprint_root_sha256"
    ]
    assert first["oracle_audit_contract"][
        "target_hidden_center_used_by_algorithm"
    ] is False
    assert first["mechanism_check"][
        "raw_sobol_median_non_dc_energy_below_source"
    ] is True
