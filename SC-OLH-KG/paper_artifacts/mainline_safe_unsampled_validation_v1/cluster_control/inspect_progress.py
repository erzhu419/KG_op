"""Small read-only Stage73 progress counts; no scientific scores or raw data."""
from pathlib import Path
import json
import shlex
import sys

root=Path('/home/erzhu419/mine_code/KG_op/SC-OLH-KG')
control=root/'paper_artifacts/mainline_safe_unsampled_validation_v1/cluster_control'
state=json.loads((control/'stage73_state.json').read_text())
label=state.get('active_stage','')
entry=state['jobs'].get(label,{})
report=dict(stage=label,status=state['status'],task_ids=entry.get('task_ids'),
    latest=state.get('latest_tasks'),error=state.get('error'),
    complete_stages={key:value.get('completion') for key,value in state['stages'].items()
                     if value.get('status')=='complete'})
if entry.get('plan') and (label.endswith('/workers') or label=='audit/tasks'):
    stage='audit' if label=='audit/tasks' else label.split('/')[0]
    paths=[f"paper_artifacts/mainline_safe_unsampled_validation_v1/cluster/{stage}/shard_{row['start']}_{row['end']}"
           for row in entry['plan']['plan']]
    sys.path.insert(0,'/home/erzhu419/mine_code/scheduleurm/skill')
    import scheduler
    python='/home/zhengliang01/scheduleurm_work/conda_envs/scomp-mainline-py310-20261017/bin/python'
    code="import json; from pathlib import Path; root=Path('/home/zhengliang01/scheduleurm_work/KG_op_mainline_cluster/SC-OLH-KG'); paths="+repr(paths)+"; out=[]\nfor name in paths:\n p=root/name; progress=p/'progress.json'; result=p/'result.json'; out.append(dict(path=name,progress=json.loads(progress.read_text()) if progress.exists() else None,failed=len(json.loads(result.read_text())['failed']) if result.exists() else None))\nprint(json.dumps(out))"
    rc,out,err=scheduler.run_on('node002',shlex.join([python,'-c',code]),timeout=30,check=False)
    report.update(progress=json.loads(out) if rc==0 else None,
        metadata_returncode=rc,metadata_payload_bytes=len(out.encode()))
with (control/'stage73_progress_snapshots.jsonl').open('a') as handle:
    handle.write(json.dumps(report)+'\n')
print(json.dumps(report))
