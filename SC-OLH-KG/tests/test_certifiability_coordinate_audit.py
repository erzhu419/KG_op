from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

from performance import (  # noqa: E402
    benchmark_certifiability_coordinate_audit as audit,
)
from performance.summarize_certifiability_coordinate_audit import (  # noqa: E402
    _best_coordinate,
    summarize,
)
from performance.benchmark_quality import parse_csv  # noqa: E402
from scripts import (  # noqa: E402
    submit_scolhkg_certifiability_coordinate_audit_scheduler as submit,
)


def _tiny_args(tmp_path):
    args = audit.build_parser().parse_args([
        "--heldout", "RZDT1",
        "--target-seed", "3",
        "--out", str(tmp_path / "result.json"),
        "--source-records", "16",
        "--source-replicates", "2",
        "--evaluation-pool", "24",
        "--oracle-candidate-pool", "32",
        "--hook-pool-per-source", "8",
        "--train-sizes", "10",
        "--training-policies", "random",
        "--regressors", "ridge_linear",
        "--replicate-budgets", "1,10",
    ])
    args.domains = tuple(parse_csv(args.domains))
    args.pilot_policies = tuple(parse_csv(args.pilot_policies))
    args.training_policies = tuple(parse_csv(args.training_policies))
    args.regressors = tuple(parse_csv(args.regressors))
    args.train_sizes = tuple(int(v) for v in parse_csv(args.train_sizes))
    args.replicate_budgets = tuple(
        int(v) for v in parse_csv(args.replicate_budgets))
    return args


def test_audit_keeps_strict_and_oracle_information_strata_separate(tmp_path):
    result = audit.run_audit(_tiny_args(tmp_path))
    strata = {row["fit_stratum"] for row in result["rows"]}
    assert strata == {"strict_source_frozen", "target_oracle_diagnostic"}
    strict = [
        row for row in result["rows"]
        if row["fit_stratum"] == "strict_source_frozen"
    ]
    oracle = [
        row for row in result["rows"]
        if row["fit_stratum"] == "target_oracle_diagnostic"
    ]
    assert all(not row["target_oracle_used_for_fit"] for row in strict)
    assert all(row["target_oracle_used_for_fit"] for row in oracle)
    assert all(
        row["training_candidate_pool"] == "uniform_random"
        for row in oracle
    )
    assert all(not row["promotion_eligible"] for row in result["rows"])
    assert result["leakage_contract"][
        "outer_target_excluded_from_source_model"]
    assert "1" in result["certifiability"]["uniform_random"][
        "replicate_budgets"]
    assert "provider_coordinate" not in result["coordinate_aliasing"]


def test_summary_rejects_promotion_and_emits_a_next_action(tmp_path):
    result = audit.run_audit(_tiny_args(tmp_path))
    result_dir = tmp_path / "RZDT1" / "seed3"
    result_dir.mkdir(parents=True)
    import json
    (result_dir / "result.json").write_text(json.dumps(result))
    summary = summarize(
        tmp_path,
        expected_domains=("RZDT1",),
        expected_seeds=1,
    )
    assert summary["n_valid_results"] == 1
    assert summary["promotion_eligible"] is False
    assert summary["next_action"] == "audit_incomplete"
    assert summary["completeness"]["RZDT1"]["complete"]
    assert summary["required_replication_groups"]
    assert summary["coordinate_aliasing_groups"]


def test_coordinate_ceiling_considers_every_preregistered_regressor():
    base = {
        "heldout": "QueueResourceControl",
        "coordinate": "raw+provider_risk",
        "coordinate_stratum": "domain_tuned_oracle_upper_bound",
        "fit_stratum": "target_oracle_diagnostic",
        "training_policy": "oracle_boundary_stratified",
        "target_train_count": 80,
    }
    rows = [
        {
            **base,
            "model_kind": "rbf_kernel_ridge",
            "metrics": {
                "spearman": -0.2,
                "normalized_boundary_mae": 0.20,
            },
        },
        {
            **base,
            "model_kind": "ridge_linear",
            "metrics": {
                "spearman": 0.85,
                "normalized_boundary_mae": 0.30,
            },
        },
    ]
    selected = _best_coordinate(
        rows,
        "QueueResourceControl",
        "domain_tuned_oracle_upper_bound",
    )
    assert selected["model_kind"] == "ridge_linear"


def test_scheduler_shards_one_domain_seed_per_twelve_core_task(tmp_path):
    args = submit.build_parser().parse_args([
        "--deploy", str(tmp_path / "deploy"),
        "--run-id", "audit_test",
        "--n-seeds", "2",
        "--dry-run",
    ])
    specs, grid, domains = submit.build_specs(args)
    assert len(grid) == 1
    assert len(specs) == domains * 2
    assert all(spec["cpu"] == 12 for spec in specs)
    assert all(spec["require_node"].startswith("node00") for spec in specs)
    assert all("benchmark_certifiability_coordinate_audit.py" in spec["cmd"]
               for spec in specs)
    assert all("OPENBLAS_NUM_THREADS=12" in spec["cmd"] for spec in specs)
