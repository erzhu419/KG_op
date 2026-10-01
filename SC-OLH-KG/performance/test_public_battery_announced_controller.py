#!/usr/bin/env python3
"""Run the affected policy regressions with separate physics/projection accounting."""
import argparse
from collections import Counter
from functools import wraps
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
from problems import public_battery_absolute_dispatch, public_battery_announced_nomination
from problems import public_battery_causal_service, public_battery_pending_availability
from problems import public_battery_receipt_forecast, public_battery_visibility


def run(out):
    counts, passed, failed = Counter(), [], []
    original_path = public_battery_visibility.trajectory

    def trajectory_counted(*args, **kwargs):
        counts["test_trajectory_evaluations"] += 1
        return original_path(*args, **kwargs)

    for module in list(sys.modules.values()):
        if (getattr(module, "__name__", "").startswith("problems.")
                and getattr(module, "trajectory", None) is original_path):
            module.trajectory = trajectory_counted

    def counted_nomination(original, name):
        @wraps(original)
        def wrapper(*args, **kwargs):
            counts[name] += 1
            command, info = original(*args, **kwargs)
            counts["test_scalar_stock_projection_evaluations"] += info.get("scalar_stock_projection_evaluations", 0)
            counts["test_announced_nomination_safety_trajectory_evaluations"] += info.get("reference_safety_path_evaluations", 0)
            return command, info
        return wrapper

    for name in ("nominate_announced", "nominate_announced_pair"):
        original = getattr(public_battery_announced_nomination, name)
        wrapper = counted_nomination(original, "test_" + name + "_calls")
        setattr(public_battery_announced_nomination, name, wrapper)
        if name == "nominate_announced":
            public_battery_causal_service.nominate_announced = wrapper
    original_projection = public_battery_announced_nomination.tail_stock_projection

    def projection_counted(*args, **kwargs):
        counts["test_direct_scalar_stock_projection_evaluations"] += 2
        counts["test_scalar_stock_projection_evaluations"] += 2
        return original_projection(*args, **kwargs)

    public_battery_announced_nomination.tail_stock_projection = projection_counted
    original_sim = public_battery_causal_service.simulate_service

    def sim_counted(*args, **kwargs):
        counts["test_short_simulation_calls"] += 1
        result = original_sim(*args, **kwargs)
        counts["test_short_simulation_minute_physics_evaluations"] += result[0]["minute_physics_evaluations"]
        counts["test_short_simulation_envelope_path_evaluations"] += result[0]["envelope_path_evaluations"]
        return result

    public_battery_causal_service.simulate_service = sim_counted

    class Accounting:
        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                (passed if report.passed else failed).append(report.nodeid)

        def pytest_sessionfinish(self, session, exitstatus):
            out.mkdir(parents=True, exist_ok=True)
            data = {"pytest_exit_code": int(exitstatus), "passed_tests": len(passed), "failed_tests": failed,
                    **dict(counts), "scope": "new announced policy and modified shared stock solver/service engine regressions"}
            (out / "unit_test_accounting.json").write_text(json.dumps(data, indent=2) + "\n")

    tests = ["announced_controller", "announced_nomination", "causal_service", "absolute_dispatch", "receipt_forecast"]
    return pytest.main(["-q", *[str(ROOT / f"tests/test_public_battery_{name}.py") for name in tests]], plugins=[Accounting()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="separate accounting folder for this test run")
    sys.exit(run(parser.parse_args().output))
