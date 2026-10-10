"""Read only the compact inputs needed for the existing V6 replay cohort."""
import base64
import gzip
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "paper_artifacts/meta_prior_calibration_replay_a_20261003"


def main():
    plan = json.loads((ROOT / "performance/meta_prior_calibration_plan.json").read_text())
    deployment = json.loads((ROOT / "paper_artifacts/meta_prior_total_budget_v6_20261002/deployment.json").read_text())
    spec = {**plan["stage_a"], "results_root": deployment["remote_results_root"]}
    # This remote command only reads archived result.json files; it fits no model.
    code = """
import base64,gzip,itertools,json
from pathlib import Path
spec=json.loads(SPEC)
rows=[]
keys=('seed','regime','difficulty','arm','allocation','queries','initial_design',
      'recommendation','initial_recommendation','predicted_verified_reward',
      'predicted_verifier_power','true_verifier_power','expected_verified_reward',
      'final_model_state')
for seed,regime,arm in itertools.product(spec['seeds'],spec['regimes'],spec['priors']):
    name=f"seed{seed}_{regime}_{spec['difficulty']}_{arm}_{spec['allocation']}"
    data=json.loads((Path(spec['results_root'])/name/'result.json').read_text())
    row=data['rows'][0]
    rows.append({'row':{k:row[k] for k in keys},'source_info':[
        {k:v for k,v in h.items() if k in ('selected_indices','source_function_variance')}
        for h in data['source_info']]})
payload={'cohort':spec['cohort'],'records':rows}
packed=gzip.compress(json.dumps(payload,separators=(',',':'),allow_nan=False).encode())
print(base64.b64encode(packed).decode())
""".replace("SPEC", repr(json.dumps(spec)))
    sys.path.insert(0, "/home/erzhu419/mine_code/scheduleurm/skill")
    import scheduler
    command = shlex.join(["env", "LC_ALL=C", "LANG=C",
        "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python", "-c", code])
    rc, out, err = scheduler.run_on("node006", command, timeout=55, check=False)
    if rc:
        raise RuntimeError(err)
    packed = base64.b64decode(out.strip())
    data = json.loads(gzip.decompress(packed))
    if len(data["records"]) != spec["existing_paths"]:
        raise ValueError("replay input cohort is incomplete")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "inputs.json.gz").write_bytes(packed)
    report = {"records": len(data["records"]), "compressed_bytes": len(packed),
              "base64_wire_bytes": len(out.strip()), "server_experiment_jobs": 0,
              "source": deployment["remote_results_root"], "node": "node006"}
    (ARTIFACTS / "collection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
