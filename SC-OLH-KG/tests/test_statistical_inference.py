from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.statistical_inference import (  # noqa: E402
    apply_holm_family,
    bootstrap_mean_ci,
    holm_adjust,
)


def test_holm_adjustment_is_step_down_and_order_preserving():
    assert holm_adjust([0.01, 0.04, 0.20]) == [0.03, 0.08, 0.2]


def test_holm_families_are_corrected_separately():
    rows = [
        {"family": "a", "p": 0.01},
        {"family": "a", "p": 0.04},
        {"family": "b", "p": 0.02},
    ]
    apply_holm_family(rows, pvalue_field="p", family_field="family")
    assert [row["p_holm"] for row in rows] == [0.02, 0.04, 0.02]
    assert [row["holm_family_size"] for row in rows] == [2, 2, 1]


def test_bootstrap_mean_uses_supplied_independent_units():
    interval = bootstrap_mean_ci([1.0], seed=7)
    assert interval == [1.0, 1.0]
    assert bootstrap_mean_ci([], seed=7) is None
