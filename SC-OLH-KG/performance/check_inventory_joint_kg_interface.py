"""One cached posterior: numerical integration and cost, without outcomes."""

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.joint_verification_kg import joint_verification_kg
from performance.run_inventory_joint_kg import prepare, models, update, moments, measurement_noise


if __name__ == "__main__":
    output = ROOT / "paper_artifacts/inventory_physical_risk_20261004/joint_kg_interface.json"
    if output.exists():
        print("Existing cached numerical check retained.")
        sys.exit(0)
    start = time.process_time()
    prepared = prepare()
    old = json.loads((ROOT / "paper_artifacts/inventory_physical_risk_20261004/closed_loop.json").read_text())
    cell = old["rows"][0]
    f, joint = models(prepared, "risk")
    for r in cell["search_path"][:8]:
        update(prepared, f, joint, r["index"], np.asarray(r["response"]), r["noise"], 12)
    state = moments(prepared, f, joint, 1.)
    noise = measurement_noise(prepared, joint, 12)
    results = {}
    for order in (3, 5):
        before = time.process_time()
        scores = joint_verification_kg(*state, noise, old["protocol"], order=order)
        results[str(order)] = {"scores": scores.tolist(), "choice": int(np.argmax(scores)),
                              "kg_cpu_seconds": time.process_time() - before}
    a, b = np.asarray(results["3"]["scores"]), np.asarray(results["5"]["scores"])
    result = {"purpose": "One cached posterior, public source priors and paid labels only; numerical check, no oracle or outcomes",
        "law_seed": cell["law_seed"], "step": 8, "results": results,
        "max_score_difference": float(np.max(np.abs(a-b))),
        "five_node_value_gap_at_three_node_choice": float(b.max()-b[int(np.argmax(a))]),
        "new_episode_calls": 0, "local_cpu_seconds": time.process_time()-start}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "results"}))
    print(json.dumps({k:{name:value for name,value in v.items() if name != "scores"} for k,v in results.items()}))
