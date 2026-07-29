import json

from performance.aggregate_proposal_coverage_audits import aggregate


DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)


def _write_audit(path, domain, *, global_bound=False):
    payload = {
        "heldout_target_domain": domain,
        "source_archive_fingerprint": f"archive-{domain}",
        "n0": 10,
        "unique_design_fingerprint_count": 1,
        "finite_sample_rank_theorem_audit": {
            "theorem_conditions_hold": False,
        },
        "geometric_atlas_theorem_audit": {
            "source_support_atlas_cover_radius": 0.04,
            "best_safe_center": {
                "source_support_shift": 0.01,
                "finite_library_safe_radius": 0.10,
                "coverage_slack": 0.05,
            },
            "finite_library_theorem_conditions_hold": True,
            "observed_atlas_contains_feasible": True,
            "target_truth_used_post_run_only": True,
            "target_truth_used_for_proposal_or_selection": False,
            "lipschitz_audit": {
                "global_lipschitz_upper_bound_certified": global_bound,
                "global_condition_status": (
                    "certified"
                    if global_bound
                    else "requires_problem_or_simulator_bound"
                ),
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_aggregate_marks_unproven_global_bound_as_conditional(tmp_path):
    paths = [
        _write_audit(tmp_path / f"{index}.json", domain)
        for index, domain in enumerate(DOMAINS)
    ]
    payload = aggregate(paths)
    assert payload["status"] == (
        "complete_with_conditional_global_bound")
    assert payload["finite_library_condition_pass_count"] == 3
    assert payload["global_lipschitz_certified_count"] == 0
    assert payload["global_theorem_claim_mode"] == "conditional_theorem_only"
    assert payload["unconditional_global_coverage_claim_allowed"] is False


def test_aggregate_allows_global_claim_only_when_all_bounds_are_certified(
    tmp_path,
):
    paths = [
        _write_audit(
            tmp_path / f"{index}.json",
            domain,
            global_bound=True,
        )
        for index, domain in enumerate(DOMAINS)
    ]
    payload = aggregate(paths)
    assert payload["status"] == "complete"
    assert payload["global_lipschitz_certified_count"] == 3
    assert payload["unconditional_global_coverage_claim_allowed"] is True
