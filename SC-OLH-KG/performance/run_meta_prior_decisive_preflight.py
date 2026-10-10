"""First new matched M1/control pair; one local thread, no full pilot dispatch."""
import json
from pathlib import Path
import resource
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.run_meta_prior_total_budget import run_shard


def main():
    start = time.process_time()
    spec = json.loads((ROOT / "performance/meta_prior_decisive_pilot_protocol.json").read_text())
    cap = spec["runtime"]["first_pair_cpu_cap_seconds"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def stop(signum, frame):
        raise TimeoutError("M1 first-pair CPU limit reached")
    signal.signal(signal.SIGXCPU, stop)
    artifacts = ROOT / "paper_artifacts" / spec["stage"]
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    pair = spec["runtime"]["first_pair"]
    result = {"stage": spec["stage"], "role": "single_new_task_preflight_not_confirmation",
        "complete_pair": False, "full_pilot_complete": False, "rows": [], "server_jobs": 0, "server_pulls": 0}
    try:
        for arm in pair["arms"]:
            before = time.process_time()
            output = run_shard(pair["seed"], pair["regime"], "binding", arm, pair["allocation"], spec)
            row = {**output["rows"][0], "source_info": output["source_info"],
                   "local_cpu_seconds": time.process_time() - before}
            result["rows"].append(row)
            (artifacts / f"first_pair_{arm}.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"arm": arm, "cpu_seconds": row["local_cpu_seconds"]}), flush=True)
        left, right = result["rows"]
        if left["initial_design"] != right["initial_design"]:
            raise ValueError("M1 matched pair initial designs differ")
        result["complete_pair"] = True
    except TimeoutError as error:
        result["stop_reason"] = str(error)
    result["cpu_seconds"] = time.process_time() - start
    result["remaining_cells_launched"] = 0
    (artifacts / "preflight.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete_pair": result["complete_pair"], "cpu_seconds": result["cpu_seconds"],
                      "full_pilot_complete": False}), flush=True)


if __name__ == "__main__":
    main()
