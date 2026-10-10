"""Source-only posterior group selector under the preserved outer holdouts."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_inventory_mean_group import run
from representation.inventory_mean_prior import fit_inventory_posterior_group_prior


if __name__ == "__main__":
    output = ROOT / "paper_artifacts/inventory_physical_risk_20261004/mean_posterior_group.json"
    if output.exists():
        print("Existing posterior group comparison retained.")
        sys.exit(0)
    p = json.loads((ROOT / "performance/inventory_mean_posterior_group_protocol.json").read_text())
    result = run(p, fit_inventory_posterior_group_prior)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": result["summary"], "selected_groups": [r["diagnostics"]["billing_block_selected"] for r in result["all_source_priors"]],
                      "selected_residuals": [r["discrepancy_variance"] for r in result["all_source_priors"]],
                      "local_cpu_seconds": result["local_cpu_seconds"]}))
