from performance.analyze_final_dimension_evidence import (
    DOMAINS,
    PROPOSAL,
    SAAS,
    SPECS,
    analyze,
)


def _record(dimension, budget, track, method, domain, seed, commit):
    return {
        "track_id": track,
        "method_identity": method,
        "domain": domain,
        "seed": seed,
        "target_dimension": dimension,
        "status": "ok",
        "source_calls": 384,
        "target_search_calls": budget,
        "target_verification_calls": 80,
        "true_feasible": True,
        "terminal_certified": True,
        "false_certificate": False,
        "feasible_regret": 0.01,
        "source_archive_fingerprint": f"source-{domain}",
        "initial_design_fingerprint": f"initial-{dimension}-{domain}-{seed}",
        "problem_contract_fingerprint": f"problem-{dimension}-{domain}",
        "verifier_signature": "verifier",
        "execution_repository_commit": commit,
    }


def _audit():
    records = []
    for dimension, spec in SPECS.items():
        for budget, (track, method) in spec["cells"].items():
            for domain in DOMAINS:
                for seed in spec["seeds"]:
                    records.append(_record(
                        dimension,
                        budget,
                        track,
                        method,
                        domain,
                        seed,
                        spec["required_execution_commit"],
                    ))
    return {
        "status": "pass",
        "registry_id": "registry",
        "records": records,
    }


def test_final_dimension_evidence_uses_stratified_seed_roles():
    result = analyze(_audit())
    assert result["status"] == "complete"
    assert result["headline_seed_counts"] == {"1000": 20, "10000": 10}
    assert result["exploratory_seed_counts"] == {"200": 5}
    assert result["cells"]["d10000_N40"][
        "dimension_over_target_search_calls"
    ] == 250.0
    assert result["all_release_rows_frozen"] is True
    assert result["d200_is_descriptive_not_confirmatory"] is True


def test_final_dimension_evidence_rejects_a_missing_cell():
    audit = _audit()
    audit["records"].pop()
    result = analyze(audit)
    assert result["status"] == "incomplete"
    assert result["missing_keys"]


def test_final_dimension_evidence_rejects_false_certification():
    audit = _audit()
    row = next(
        row for row in audit["records"]
        if row["target_dimension"] == 1000 and row["method_identity"] == SAAS
    )
    row["false_certificate"] = True
    result = analyze(audit)
    assert result["status"] == "incomplete"
    assert any("false certificates" in failure for failure in result["failures"])
