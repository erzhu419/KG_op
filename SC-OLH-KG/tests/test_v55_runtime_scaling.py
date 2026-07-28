from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "performance/analyze_v55_runtime_scaling.py"
SPEC = importlib.util.spec_from_file_location("v55_runtime_scaling", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_runtime_scaling_loader_preserves_score_and_worker_contract(tmp_path):
    root = tmp_path / "run"
    path = root / "v55_mc128/shock0/FactorShock/seed0/result.json"
    path.parent.mkdir(parents=True)
    trace = {
        "x_fingerprint": "selected",
        "exact_kg_active_action_fingerprints": ["a", "b"],
    }
    for index, field in enumerate(MODULE.SCORE_FIELDS):
        trace[field] = [float(index), float(index + 1)]
    path.write_text(json.dumps({
        "config": {"exact_kg_jobs": 24},
        "rows": [{
            "heldout": "FactorShock",
            "algorithm_time_sec": 3.0,
            "stage_times": {"t_kg_compute": {"total": 2.0}},
            "online_action_trace": [trace],
        }],
    }))

    records = MODULE.load_run(24, root)
    record = records["FactorShock"]
    assert record["exact_jobs"] == 24
    assert record["selected_fingerprint"] == "selected"
    np.testing.assert_allclose(
        record["scores"][MODULE.SCORE_FIELDS[-1]], [5.0, 6.0])
