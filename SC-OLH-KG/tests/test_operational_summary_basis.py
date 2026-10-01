from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pytest
from problems.rzdt import InventorySupplyChainProblem, QueueResourceControlProblem


@pytest.mark.parametrize("problem_class", [InventorySupplyChainProblem, QueueResourceControlProblem])
def test_surrogate_distinguishes_the_reported_safety_or_smoothing_alias(problem_class):
    problem = problem_class(d=6, L=100)
    x, y = (50, 50, 50, 50, 30, 30), (50, 50, 50, 50, 70, 70)
    assert not np.isclose(problem.true_objectives(x)[2], problem.true_objectives(y)[2])
    assert not np.allclose(problem.gpr_basis_map().features(x), problem.gpr_basis_map().features(y))
