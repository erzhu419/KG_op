"""Real source-prior restoration and cached raw initialization replay only."""

import gc
import json
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

START_CPU = time.process_time()
START_WALL = time.monotonic()
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))


def stop(_signal, _frame):
    raise TimeoutError("30-second single-thread CPU smoke limit")


signal.signal(signal.SIGPROF, stop)
signal.setitimer(signal.ITIMER_PROF, 30.0)
result = {
    "scope": "Real 73-policy source/generic prior recovery and two cached initialization episodes",
    "new_physical_calls": 0,
    "new_verifier_calls": 0,
    "new_source_fits": 0,
    "kg_or_terminal_integration": False,
    "rows": [],
    "failures": [],
}
try:
    import numpy as np
    import scipy
    from performance.historical_common_bank import (
        common_bank_mapping, initialization_batches, replay_raw_batches,
        restore_zero_paid_prior,
    )
    from performance.historical_full_bank_development import (
        PLAN, context, read, restricted_snapshot,
    )
    from performance.prepare_historical_independent_kg import check_replayed_snapshot

    result["environment"] = {
        "executable": sys.executable, "python": sys.version.split()[0],
        "numpy": np.__version__, "scipy": scipy.__version__,
        "root": str(ROOT), "plan": PLAN,
    }
    spec = read(PLAN)
    pilot, frozen, states, _public, connection, old, bank = context(spec)
    task_id = frozen["tasks"][0]["task_id"]
    layout = frozen["banks"][0]["layout"]
    paid = next(r for r in read(spec["initializations"])["rows"] if r["task_id"] == task_id)
    frozen_bank = next(b for b in frozen["banks"] if b["layout"] == layout)
    mapping = common_bank_mapping(bank, frozen_bank)
    selected = [s for s in states if s["task_id"] == task_id and s["layout"] == layout]
    assert {s["arm"] for s in selected} == {"source_structural_mixture", "source_common_task_rbf"}
    for state in selected:
        cpu = time.process_time()
        wall = time.monotonic()
        model, diagnostics = restore_zero_paid_prior(spec, pilot, state, connection, old, len(bank))
        prior_mean, prior_cov = model.predict_joint()
        prior_weights = model.weights.copy()
        component_counts = [len(field.means) if hasattr(field, "means") else 1
                            for pair in model.components for field in pair]
        assert prior_mean.shape == (73, 4) and prior_cov.shape == (73, 4, 73, 4)
        assert np.isfinite(prior_mean).all() and np.isfinite(prior_cov).all()
        replay = replay_raw_batches(model, initialization_batches(paid), paid["tau"])
        post_mean, post_cov = model.predict_joint()
        assert np.isfinite(post_mean).all() and np.isfinite(post_cov).all()
        assert model.paid_episodes == 2 and replay["remaining_calls"] == 360
        assert all(field.candidate_count == 73 and field.paid_episodes == 2
                   for pair in model.components for field in pair)
        check_replayed_snapshot(restricted_snapshot(model, state["candidate_bank_indices"]),
            state["posterior_moment_diagnostic"], roundoff_units=146)
        unseen = np.array([i for i in range(73) if i not in paid["candidate_bank_indices"]])
        changes = post_mean[unseen] - prior_mean[unseen]
        row = {
            "state": state["name"], "task_id": task_id, "layout": layout, "arm": state["arm"],
            "candidate_count": 73, "prior_paid_episodes": 0, "replay": replay,
            "paid_policy_indices": paid["candidate_bank_indices"], "mapped_queries": mapping["queries"],
            "complete_identities": len(model.components), "components_per_head": component_counts,
            "all_prior_and_posterior_moments_finite": True, "native_eight_point_replay_passed": True,
            "unobserved_policy_count": len(unseen),
            "unobserved_mean_changes_max_abs_f_eta_f_g_eta_g": np.max(abs(changes), axis=0).tolist(),
            "unobserved_policies_with_mean_change_over_1e_12": int(np.sum(np.max(abs(changes), axis=1) > 1e-12)),
            "joint_covariance_max_absolute_change": float(np.max(abs(post_cov-prior_cov))),
            "prior_weights": prior_weights.tolist(), "posterior_weights": model.weights.tolist(),
            "restoration_diagnostics": diagnostics,
            "cpu_seconds": time.process_time()-cpu, "wall_seconds": time.monotonic()-wall,
        }
        result["rows"].append(row)
        print(json.dumps({k: row[k] for k in ("arm", "candidate_count", "complete_identities", "cpu_seconds")}), flush=True)
        del model, prior_mean, prior_cov, post_mean, post_cov
        gc.collect()
    result["passed"] = len(result["rows"]) == 2
except Exception as exc:
    result["passed"] = False
    result["failures"].append({"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
finally:
    signal.setitimer(signal.ITIMER_PROF, 0)
    result["total_cpu_seconds"] = time.process_time()-START_CPU
    result["total_wall_seconds"] = time.monotonic()-START_WALL
    result["maximum_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (OUT/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ("passed", "total_cpu_seconds", "total_wall_seconds", "maximum_rss_kib")}), flush=True)
sys.exit(0 if result["passed"] else 1)
