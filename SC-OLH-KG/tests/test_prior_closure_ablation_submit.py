import importlib.util
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/submit_scolhkg_prior_closure_ablation_scheduler.py"
SPEC = importlib.util.spec_from_file_location("prior_closure_submit", SCRIPT)
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)


def _args():
    return SimpleNamespace(
        scheduler=Path("/scheduler.py"),
        deploy=Path("/deploy"),
        manifest=Path("/deploy/SC-OLH-KG/performance/manifests/base.json"),
        heldouts=(
            "FactorShockStatePolicyRZDT1,InventorySupplyChain,"
            "QueueResourceControl"
        ),
        seed_start=0,
        n_seeds=20,
        nodes="node001,node002,node003,node004,node005,node006",
        cpu=12,
        ram_mb=32768,
        exact_jobs=12,
        dry_run=True,
    )


def test_matrix_has_four_atomic_prior_ablations_and_two_closed_contracts():
    expected = {
        "promoted_legacy",
        "closed_terminal_kg",
        "closed_all_exact",
        "drop_low_frequency",
        "drop_orthogonality",
        "drop_coefficient_sparsity",
        "drop_additivity",
        "drop_all_four",
    }
    assert set(submit.VARIANTS) == expected

    closed = submit.command(
        _args(), "audit", "closed_terminal_kg",
        submit.VARIANTS["closed_terminal_kg"])
    text = " ".join(map(str, closed))
    assert "--decision-contract-mode certified_lexicographic" in text
    assert "--finalist-replication-policy terminal_kg_1step" in text
    assert "--finalist-empirical-override off" in text
    assert "--finalist-terminal-value-mode certified_lexicographic" in text

    no_orthogonal = submit.command(
        _args(), "audit", "drop_orthogonality",
        submit.VARIANTS["drop_orthogonality"])
    assert "--no-ordered-orthogonal-coordinates" in no_orthogonal

    no_sparse = submit.command(
        _args(), "audit", "drop_coefficient_sparsity",
        submit.VARIANTS["drop_coefficient_sparsity"])
    assert "--no-ordered-adaptive-sparsity" in no_sparse
    assert "--no-ordered-group-ridge-learning" in no_sparse

    no_additive = submit.command(
        _args(), "audit", "drop_additivity",
        submit.VARIANTS["drop_additivity"])
    joined = " ".join(map(str, no_additive))
    assert "--ordered-basis-mode full_quadratic" in joined


def test_matrix_defaults_to_twenty_seeds_and_twelve_cores_per_seed():
    args = _args()
    cmd = submit.command(
        args, "audit", "promoted_legacy",
        submit.VARIANTS["promoted_legacy"])
    text = " ".join(map(str, cmd))
    assert "--n-seeds 20" in text
    assert "--cpu 12" in text
    assert "--exact-jobs 12" in text
