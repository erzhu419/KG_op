from performance.analyze_dimension_budget_frontier import analyze_rows


DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)


def _row(dimension, domain, seed, budget, *, feasible=True, regret=0.01):
    return {
        "phase": "proposal" if budget == 10 else "saas",
        "budget": budget,
        "dimension": dimension,
        "domain": domain,
        "seed": seed,
        "status": "ok",
        "true_feasible": feasible,
        "terminal_certified": feasible,
        "false_certificate": False,
        "feasible_regret": regret if feasible else float("inf"),
        "source_archive_fingerprint": f"archive-{domain}",
        "initial_design_fingerprint": f"design-{dimension}-{domain}-{seed}",
        "problem_contract_fingerprint": f"problem-{dimension}-{domain}",
        "verifier_signature": "verifier",
    }


def _matrix():
    rows = []
    for dimension in (200, 1000):
        for domain in DOMAINS:
            for seed in (80, 81):
                rows.append(_row(dimension, domain, seed, 10))
                for budget in (20, 40, 80):
                    rows.append(_row(
                        dimension,
                        domain,
                        seed,
                        budget,
                        regret=0.005,
                    ))
    return rows


def test_frontier_expands_only_after_complete_paired_noninferiority():
    report = analyze_rows(
        _matrix(),
        dimensions=(200, 1000),
        budgets=(20, 40, 80),
        domains=DOMAINS,
        seeds=(80, 81),
        evidence_source="unit",
    )
    assert report["status"] == "complete"
    assert report["observed_row_count"] == 48
    assert report["expand_to_20_seeds"]
    assert all(
        gate["adaptive_loss_count"] == 0
        for gate in report["gates"].values()
    )


def test_frontier_rejects_adaptive_loss_and_contract_mismatch():
    rows = _matrix()
    damaged = next(
        row for row in rows
        if row["dimension"] == 1000
        and row["domain"] == "QueueResourceControl"
        and row["seed"] == 80
        and row["budget"] == 80
    )
    damaged["true_feasible"] = False
    damaged["terminal_certified"] = False
    damaged["feasible_regret"] = float("inf")
    damaged["initial_design_fingerprint"] = "changed"
    report = analyze_rows(
        rows,
        dimensions=(200, 1000),
        budgets=(20, 40, 80),
        domains=DOMAINS,
        seeds=(80, 81),
        evidence_source="unit",
    )
    assert report["status"] == "incomplete"
    assert not report["expand_to_20_seeds"]
    assert report["contract_mismatches"]
    assert report["gates"]["d1000_N80"]["adaptive_loss_count"] == 1
