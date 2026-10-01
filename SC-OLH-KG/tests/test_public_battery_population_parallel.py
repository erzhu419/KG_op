"""Parallel ownership and resume tests; all cell simulations are mocked."""
import json

import pytest

from performance import run_public_battery_population_parallel as q
from performance.inspect_public_battery_announced_development import CALL_FIELDS


def identity(minute):
    return dict(period="2024-04-15", window_start_minute=minute, profile_id="pair", dispatch_rule="conservative_admission")


def test_remaining_work_skips_known_outcomes_and_preserves_missing_success_costs():
    protocol = {"population": {"window_start_minutes": [0, 60], "development_samples": ["a", "b"], "dispatch_rules": ["x", "y"]}}
    profiles = [("one", "constant", None), ("two", "functional", None)]
    rows = {("a", 0, "one", "x"): {"window_success": False, "cost_GBP": None},
            ("a", 0, "one", "y"): {"window_success": True, "cost_GBP": 1.},
            ("b", 0, "two", "x"): {"window_success": True, "cost_GBP": None}}
    work = q.remaining_work(protocol, profiles, rows)
    assert len(work) == 14
    assert sum(r["evaluation_origin"] == "economic_replay" for r in work) == 1
    shards = q.split_work(work, 5)
    covered = [r for s in shards for r in work[s["start"]:s["stop"]]]
    assert covered == work and len({q.p.key(r) for r in covered}) == 14
    sizes = [s["stop"] - s["start"] for s in shards]
    assert max(sizes) - min(sizes) == 1


@pytest.fixture
def state(tmp_path, monkeypatch):
    work = [{**identity(m), "profile_index": 0, "evaluation_origin": "economic_replay" if m == 0 else "new_population"} for m in (0, 60, 120)]
    base = {q.p.key(work[0]): {**identity(0), "window_success": True, "cost_GBP": None}}
    windows = {("2024-04-15", m): {"metadata": {}, "start": None, "prices": None} for m in (0, 60, 120)}
    monkeypatch.setattr(q, "STATE", dict(campaign=tmp_path, work=work, plan={"shards": q.split_work(work, 1), "shard_count": 1},
                                       baseline=base, profiles=[("pair", "constant", None)], nodes=None,
                                       windows=windows, data={"asset": {}}, contract={}))
    calls = []
    def execute(item, *args):
        calls.append(q.p.key(item))
        return {"window_success": item["window_start_minute"] == 0, **{f: 2 for f in CALL_FIELDS}}, [], []
    monkeypatch.setattr(q.p, "execute", execute)
    monkeypatch.setattr(q.p, "validate_result", lambda *a: None)
    monkeypatch.setattr(q.p, "economics", lambda r, *a: {"cost_GBP": 12. if r["window_success"] else None})
    monkeypatch.setattr(q.p, "compare_saved", lambda *a: None)
    monkeypatch.setattr(q.p, "failure_context", lambda r, *a: None if r["window_success"] else {"failure": "mocked service shortfall"})
    return tmp_path, work, base, calls


def test_completed_shard_resumes_without_calls_and_merges_replay_cost_once(state):
    root, work, base, calls = state
    assert q.run_shard(0)["completed_cells"] == 3
    assert len(calls) == 3
    q.run_shard(0)
    assert len(calls) == 3
    rows, replays, executions, failures, seen = dict(base), [], [], [], set()
    q.merge_shard(root, 0, work, rows, replays, executions, failures, seen)
    assert len(rows) == 3 and len(replays) == 1 and len(executions) == 3 and len(failures) == 2
    assert rows[q.p.key(work[0])]["cost_GBP"] == 12.
    assert all(rows[q.p.key(r)]["cost_GBP"] is None for r in work[1:])
    with pytest.raises(ValueError, match="overlaps"):
        q.merge_shard(root, 0, work, rows, replays, executions, failures, seen)


def test_interrupted_started_cell_blocks_silent_rerun(state):
    root, work, _, calls = state
    out = root / "shards/00000"
    out.mkdir(parents=True)
    q.p.append(out, "started.jsonl", work[0])
    with pytest.raises(ValueError, match="interrupted"):
        q.run_shard(0)
    assert calls == []


def test_missing_validated_result_blocks_aggregation(state):
    root, work, base, _ = state
    q.run_shard(0)
    (root / "shards/00000/cells.jsonl").write_text("")
    with pytest.raises(ValueError, match="unresolved"):
        q.merge_shard(root, 0, work, dict(base), [], [], [], set())


def test_real_process_pool_owns_disjoint_shards_on_distinct_physical_cpus(state, monkeypatch):
    root, work, _, _ = state
    q.STATE["plan"] = {"shards": q.split_work(work, 3), "shard_count": 3}
    monkeypatch.setattr(q, "load_state", lambda campaign: None)
    q.run(root, 0, 3, 2)
    status = json.loads((root / "node_task_00000_00003.json").read_text())
    assert status["status"] == "complete" and status["workers"] == 2
    assert len(set(status["physical_cpus"])) >= 2
    records = [r for folder in (root / "shards").iterdir() for r in q.p.lines(folder / "executions.jsonl")]
    assert len(records) == len({q.p.key(r) for r in records}) == 3
