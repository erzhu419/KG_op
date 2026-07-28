from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "performance"))

from evaluate_dimension_holdout_gate import evaluate_gate  # noqa: E402


DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)


def _rows():
    rows = []
    for mode in ("proposal_only", "joint"):
        for proposal in (
            "risk_objective_atlas",
            "risk_coordinate_atlas",
            "rank_spanning",
        ):
            for profile in ("none", "orthogonality_only"):
                for domain in DOMAINS:
                    for seed in range(5):
                        if profile == "orthogonality_only":
                            regret = 0.10 if proposal == "risk_objective_atlas" else 0.15
                        else:
                            regret = 0.20 if proposal == "risk_objective_atlas" else 0.25
                        rows.append({
                            "run_id": "dimension-gate",
                            "variant": (
                                f"causal_prior_v2/{mode}/{proposal}/{profile}"),
                            "status": "ok",
                            "domain": domain,
                            "seed": seed,
                            "d": 200,
                            "N": 20,
                            "n0": 10,
                            "source_calls": 384,
                            "source_archive_fingerprint": "same-archive",
                            "initial_design_fingerprint": (
                                f"{proposal}-{profile}-{domain}-{seed}"),
                            "initial_has_true_feasible": True,
                            "initial_best_feasible_regret": regret,
                            "true_feasible": True,
                            "feasible_regret": regret,
                            "adaptive_loss": False,
                        })
    return rows


def test_dimension_gate_promotes_safe_objective_improving_profile():
    report = evaluate_gate(
        _rows(),
        challenger="risk_objective_atlas",
        references=("risk_coordinate_atlas", "rank_spanning"),
        profiles=("orthogonality_only",),
        domains=DOMAINS,
        seeds=range(5),
    )

    assert report["missing_cell_count"] == 0
    assert report["promoted_profiles"] == ["orthogonality_only"]
    profile = report["profiles_result"]["orthogonality_only"]
    assert profile["safety"]["overall_feasible"] == 15
    assert profile["structural_effect_vs_none"]["conditional_regret"]["wins"] == 15
    assert all(
        item["pass"] for item in profile["proposal_comparisons"].values())


def test_dimension_gate_reports_missing_challenger_cell():
    rows = _rows()
    rows = [
        row for row in rows
        if not (
            row["variant"] == (
                "causal_prior_v2/joint/risk_objective_atlas/"
                "orthogonality_only")
            and row["domain"] == DOMAINS[0]
            and row["seed"] == 0
        )
    ]
    report = evaluate_gate(
        rows,
        challenger="risk_objective_atlas",
        references=("risk_coordinate_atlas", "rank_spanning"),
        profiles=("orthogonality_only",),
        domains=DOMAINS,
        seeds=range(5),
    )

    assert report["missing_cell_count"] == 1
    assert report["promoted_profiles"] == []
