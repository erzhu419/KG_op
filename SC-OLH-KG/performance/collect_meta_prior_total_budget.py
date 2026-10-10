"""Finish V6 accounting and pull its compact summary after the scheduler batch."""
import gzip
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "paper_artifacts/meta_prior_total_budget_v6_20261002"
SCHEDULER = Path("/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py")
REMOTE_PYTHON = "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python"


def main():
    deployment = json.loads((ARTIFACTS / "deployment.json").read_text())
    subprocess.run([sys.executable, str(SCHEDULER), "wait-for", "--signature",
        "KG_op/" + deployment["stage"] + "/*", "--poll", "30", "--timeout", "0", "--verbose"], check=True)
    sys.path.insert(0, str(SCHEDULER.parent))
    import scheduler
    code_root = deployment["remote_code_root"]
    results = deployment["remote_results_root"]
    summary = results + "/summary.json"
    cmd = shlex.join(["env", "LC_ALL=C", "LANG=C", "OPENBLAS_NUM_THREADS=1",
        REMOTE_PYTHON, code_root + "/performance/summarize_meta_prior_total_budget.py",
        "--results-dir", results, "--protocol", code_root + "/performance/meta_prior_total_budget_protocol.json",
        "--output", summary])
    rc, out, err = scheduler.run_on("node006", cmd, timeout=55, check=False)
    print(out, flush=True)
    if rc:
        raise RuntimeError(err)
    subprocess.run([sys.executable, str(SCHEDULER.parent / "transport_exec.py"), "--bulk", "--",
        "rsync", "-a", "-e", "ssh -o ConnectTimeout=30 -o BatchMode=yes -J jtl110gpu",
        "zhengliang01@202.197.46.16:" + summary + ".gz", str(ARTIFACTS / "summary.json.gz")], check=True)
    compressed = ARTIFACTS / "summary.json.gz"
    data = gzip.decompress(compressed.read_bytes())
    result = json.loads(data)
    (ARTIFACTS / "summary.json").write_bytes(data)
    (ARTIFACTS / "completion.json").write_text(json.dumps({
        "stage": deployment["stage"], "complete_cells": result["complete_cells"],
        "independent_seed_blocks": result["independent_seed_blocks"],
        "fixed_budget_and_source_pairing_passed": result["fixed_budget_and_source_pairing_passed"],
        "false_verified": result["false_verified"], "pulled_bytes": compressed.stat().st_size,
        "interpretation": "Full numerical results available; scientific interpretation remains to be written."
    }, indent=2) + "\n")
    print("V6 COMPLETE: compact summary collected", flush=True)


if __name__ == "__main__":
    main()
