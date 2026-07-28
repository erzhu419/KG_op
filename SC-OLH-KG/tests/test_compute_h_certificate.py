import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.kg import (  # noqa: E402
    compute_h,
    compute_h_certificate,
    validate_h_certificate,
)


def assert_stack_snapshot_valid(testcase, snapshot):
    slopes = snapshot["hull_slopes"]
    cuts = snapshot["cuts"]
    testcase.assertEqual(len(slopes), len(cuts))
    testcase.assertEqual(
        len(snapshot["hull_indices"]),
        len(slopes),
    )
    testcase.assertEqual(
        len(snapshot["hull_intercepts"]),
        len(slopes),
    )
    for left, right in zip(slopes, slopes[1:]):
        testcase.assertLess(left, right)
    for left, right in zip(cuts, cuts[1:]):
        testcase.assertLess(left, right)
    if cuts:
        testcase.assertTrue(np.isneginf(cuts[0]))


class ComputeHCertificateTests(unittest.TestCase):
    def test_certificate_validates_random_line_envelopes(self):
        rng = np.random.default_rng(123)
        for n_lines in (2, 3, 8, 17):
            for _ in range(20):
                a = rng.normal(size=n_lines)
                b = rng.normal(size=n_lines)
                cert = compute_h_certificate(a, b)
                check = validate_h_certificate(a, b, cert)
                self.assertTrue(check["valid"], check["errors"])
                self.assertAlmostEqual(cert.h_value, compute_h(a, b), places=12)
                self.assertGreaterEqual(cert.h_value, 0.0)
                self.assertGreater(len(cert.trace), 0)
                for snapshot in cert.trace:
                    assert_stack_snapshot_valid(self, snapshot)

    def test_certificate_handles_duplicate_slopes(self):
        a = np.array([0.0, 2.0, -1.0, 1.0])
        b = np.array([0.5, 0.5, 1.0, -0.25])
        cert = compute_h_certificate(a, b)
        check = validate_h_certificate(a, b, cert)
        self.assertTrue(check["valid"], check["errors"])
        self.assertIn(1, cert.hull_indices)
        self.assertNotIn(0, cert.hull_indices)

    def test_trace_records_pop_and_break_steps(self):
        a = np.array([0.0, -1.0, -0.5])
        b = np.array([0.0, 1.0, 2.0])
        cert = compute_h_certificate(a, b)
        actions = [row["action"] for row in cert.trace]
        self.assertIn("pop", actions)
        self.assertIn("break", actions)
        self.assertEqual(actions[-1], "push")
        for snapshot in cert.trace:
            assert_stack_snapshot_valid(self, snapshot)

    def test_validator_rejects_corrupted_cut(self):
        a = np.array([0.0, 1.0, -0.5])
        b = np.array([-1.0, 0.0, 1.0])
        cert = compute_h_certificate(a, b)
        corrupted = cert.__class__(
            input_size=cert.input_size,
            baseline=cert.baseline,
            envelope_expectation=cert.envelope_expectation,
            h_value=cert.h_value,
            hull_indices=cert.hull_indices,
            hull_intercepts=cert.hull_intercepts,
            hull_slopes=cert.hull_slopes,
            cuts=tuple(0.0 for _ in cert.cuts),
            prob_masses=cert.prob_masses,
            first_moments=cert.first_moments,
            contributions=cert.contributions,
        )
        check = validate_h_certificate(a, b, corrupted)
        self.assertFalse(check["valid"])


if __name__ == "__main__":
    unittest.main()
