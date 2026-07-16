from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "performance"))

from evaluate_low_frequency_support_gate import evaluate_gate  # noqa: E402


DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)


def _rows(*, joint_low_frequency_feasible=True):
    rows = []
    for mode in ("proposal_only", "joint"):
        for profile in ("none", "low_frequency_only"):
            for domain in DOMAINS:
                for seed in range(5):
                    feasible = profile == "low_frequency_only"
                    if mode == "joint" and profile == "low_frequency_only":
                        feasible = joint_low_frequency_feasible
                    rows.append({
                        "run_id": "low-frequency-gate",
                        "variant": (
                            "causal_prior_v2/"
                            f"{mode}/risk_objective_atlas/{profile}"
                        ),
                        "status": "ok",
                        "domain": domain,
                        "seed": seed,
                        "d": 200,
                        "N": 20,
                        "n0": 10,
                        "source_calls": 384,
                        "source_archive_fingerprint": "same-archive",
                        "initial_design_fingerprint": (
                            f"{mode}-{profile}-{domain}-{seed}"
                        ),
                        "initial_has_true_feasible": feasible,
                        "initial_best_feasible_regret": 0.1 if feasible else None,
                        "true_feasible": feasible,
                        "feasible_regret": 0.1 if feasible else None,
                        "adaptive_loss": False,
                    })
    return rows


def test_low_frequency_gate_promotes_modes_separately():
    report = evaluate_gate(_rows(), domains=DOMAINS, seeds=range(5))

    assert report["missing_cell_count"] == 0
    assert report["archive_contract_pass"] is True
    assert report["expected_source_calls"] == 384
    assert report["source_call_values"] == [384]
    assert all(
        values == ["same-archive"]
        for values in report["source_archive_fingerprints"].values()
    )
    assert report["promoted_modes"] == ["proposal_only", "joint"]
    for mode in report["promoted_modes"]:
        result = report["mode_results"][mode]
        assert result["safety"]["overall_feasible"] == 15
        assert result["paired_effect_vs_none"]["feasibility"]["net"] == 15
        assert result["paired_effect_vs_none"]["conditional_regret"]["net"] == 0


def test_low_frequency_gate_does_not_hide_joint_failure():
    report = evaluate_gate(
        _rows(joint_low_frequency_feasible=False),
        domains=DOMAINS,
        seeds=range(5),
    )

    assert report["promoted_modes"] == ["proposal_only"]
    assert report["mode_results"]["proposal_only"]["promote"] is True
    assert report["mode_results"]["joint"]["promote"] is False


def test_low_frequency_gate_reports_missing_matched_cell():
    rows = _rows()
    rows = [
        row for row in rows
        if not (
            row["variant"] == (
                "causal_prior_v2/proposal_only/risk_objective_atlas/none"
            )
            and row["domain"] == DOMAINS[0]
            and row["seed"] == 0
        )
    ]
    report = evaluate_gate(rows, domains=DOMAINS, seeds=range(5))

    assert report["missing_cell_count"] == 1
    assert report["mode_results"]["proposal_only"]["promote"] is False


def test_low_frequency_gate_rejects_mixed_source_archives():
    rows = _rows()
    rows[0]["source_archive_fingerprint"] = "different-archive"
    report = evaluate_gate(rows, domains=DOMAINS, seeds=range(5))

    assert report["archive_contract_pass"] is False
    assert report["promoted_modes"] == []
