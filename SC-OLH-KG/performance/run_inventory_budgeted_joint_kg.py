"""Run the separately frozen budget-aware joint-KG development pilot."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.run_inventory_joint_kg import run


if __name__ == "__main__":
    p = json.loads((ROOT / "performance/inventory_budgeted_joint_kg_protocol.json").read_text())
    output = ROOT / "paper_artifacts/inventory_physical_risk_20261004/budgeted_joint_kg.json"
    result = run(p, output)
    print(json.dumps({"complete": result["complete"], "summary": result["summary"], "local_cpu_seconds": result["local_cpu_seconds"]}))
