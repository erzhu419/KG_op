"""The grid audit has separate meters for held-out and verification-law draws."""
import numpy as np
import pytest

from v2_or.contracts import BaseDynamicORProblem
from v2_or.simulators.opsd_storage import OPSDGridStorageBenchmark


def test_grid_audit_meters_draws_by_pool_and_records_failed_batches(monkeypatch):
    grid = OPSDGridStorageBenchmark.__new__(OPSDGridStorageBenchmark)
    BaseDynamicORProblem.__init__(grid)
    grid.d, grid.buffer_hours, grid.clamp_limit, grid.tail_clamped_fraction = 1, 24, 0.2, 0.0
    grid.call_limits = {"audit": 2000, "verification": 2512}
    draws = []

    def simulate(policy, reps, rng, split):
        draws.append((split, reps))
        return np.zeros((reps, 2))

    monkeypatch.setattr(grid, "_simulate_raw", simulate)
    for split in ("audit", "verification"):
        result = grid.audit_ground_truth([0.5], n_eval=2000, split=split)
        assert result["n_audit_samples"] == 2000
        assert result["audit_split"] == split
    assert grid.audit_calls == grid.verification_calls == 2000
    assert draws == [(s, n) for s in ("audit", "verification") for n in (512, 512, 512, 464)]

    def fail(policy, reps, rng, split):
        raise RuntimeError("interrupted simulator batch")

    monkeypatch.setattr(grid, "_simulate_raw", fail)
    with pytest.raises(RuntimeError, match="interrupted simulator batch"):
        grid.audit_ground_truth([0.5], n_eval=2000, split="verification")
    assert grid.attempted_calls == {"audit": 2000, "verification": 2512}
    assert grid.verification_calls == 2000
    with pytest.raises(RuntimeError, match="exceeds verification cap"):
        grid.audit_ground_truth([0.5], n_eval=2000, split="verification")
