"""Deterministic branch checks for the executed terminal safety argument."""
from itertools import product

import numpy as np
import pytest

from v2_or.verification.gated_verifier_v2 import verify_and_deploy_shortlist


class FixedVerificationBlocks:
    alpha = 0.05

    def __init__(self, gate_pass, objective):
        self.gate_pass = gate_pass
        self.objective = objective
        self.calls = []

    def simulate_replications(self, x, reps, rng, *, split):
        assert split == "verification"
        slot = int(x[0])
        self.calls.append((slot, reps))
        return np.tile((self.objective[slot], -1.0 if self.gate_pass[slot] else 1.0), (reps, 1))


@pytest.mark.parametrize("guard", [False, True])
def test_three_slot_gate_paths(guard):
    budgets = (112, 128, 140)
    roles = ("posterior_primary",
             "empirical_feasible_incumbent" if guard else "conservative_posterior_interior",
             "conservative_posterior_interior")
    shortlist = [(role, np.array([slot], dtype=float)) for slot, role in enumerate(roles)]
    for gates in product((False, True), repeat=3):
        for objective in ((1.0, 2.0, 3.0), (2.0, 1.0, 3.0), (1.0, 1.0, 3.0)):
            problem = FixedVerificationBlocks(gates, objective)
            result = verify_and_deploy_shortlist(problem, shortlist, seed=7, gate_mode="exact_binomial")
            passing = [i for i, passed in enumerate(gates) if passed]
            expected = passing[0] if passing else None
            attempts = (expected + 1) if expected is not None else 3
            if guard and (gates[0] or gates[1]):
                expected = min([i for i in (0, 1) if gates[i]], key=lambda i: (objective[i], i))
                attempts = 2
            assert result.selected_slot == expected
            assert result.certified == (expected is not None)
            assert problem.calls == list(enumerate(budgets[:attempts]))
            assert result.verification_calls_used == sum(budgets[:attempts])
            assert result.slot_deltas_configured == pytest.approx((0.025, 0.015, 0.010))
            assert all(c.critical_k == 1 for c in result.certificates)
            if expected is None:
                assert result.decision_state == "abstention"
                assert result.deployed_policy is None
            else:
                assert result.certificates[expected].certified
                assert result.deployed_policy.tolist() == [float(expected)]


@pytest.mark.parametrize("roles,allocations", [
    (("posterior_primary",), (0.05,)),
    (("posterior_primary", "empirical_feasible_incumbent"), (0.03125, 0.01875)),
])
def test_shortlist_allocations_after_deduplication(roles, allocations):
    problem = FixedVerificationBlocks((False, False), (1.0, 2.0))
    shortlist = [(role, np.array([slot], dtype=float)) for slot, role in enumerate(roles)]
    result = verify_and_deploy_shortlist(problem, shortlist, seed=7, gate_mode="exact_binomial")
    assert result.slot_deltas_configured == pytest.approx(allocations)
    assert result.verification_calls_used == sum((112, 128)[:len(roles)])
    assert result.decision_state == "abstention"
    assert result.deployed_policy is None
