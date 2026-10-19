#!/usr/bin/env python3
"""Resume Stage73's single frozen cohort through scheduler and global barriers."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]/'SC-OLH-KG'
SCHEDULER = PROJECT.parent.parent/'scheduleurm/skill/scheduler.py'
REMOTE = '/home/zhengliang01/scheduleurm_work/KG_op_mainline_cluster/SC-OLH-KG'
PYTHON = '/home/zhengliang01/scheduleurm_work/conda_envs/scomp-mainline-py310-20261017/bin/python'
NODES = [f'node{i:03d}' for i in range(1, 7)]
STAGES = ('initial', 'neighbors', 'online', 'native', 'old', 'new')
MANIFEST = 'performance/manifests/mainline_safe_unsampled_validation_v1.json'
AUDIT = 'performance.run_mainline_safe_unsampled_audit'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n')


def module(name, *arguments):
    return [PYTHON, '-u', '-m', name, *map(str, arguments)]


class Controller:
    stage_label = 'Stage73'
    signature_stage = 'stage73'

    def __init__(self, manifest=MANIFEST, poll_seconds=30):
        self.manifest = manifest
        self.plan = read(PROJECT/manifest)
        assert self.plan['name'] == 'mainline_safe_unsampled_validation_v1'
        assert len(self.plan['tasks']) == 24 and len(self.plan['replicates']) == 1
        self.output = PROJECT/self.plan['output_dir']/'cluster_control'
        self.state_path = self.output/'stage73_state.json'
        self.state = read(self.state_path) if self.state_path.exists() else dict(
            manifest=manifest, status='not_started', stages={}, jobs={}, attempts=[],
            node_pool=NODES, remote_cwd=REMOTE, remote_python=PYTHON,
            local_scientific_computation=False)
        self.poll_seconds = poll_seconds

    def update(self, **fields):
        self.state.update(fields, updated_utc=datetime.now(timezone.utc).isoformat())
        save(self.state_path, self.state)

    def capture(self, label, argv):
        result = subprocess.run(argv, cwd=PROJECT, text=True, capture_output=True)
        record = dict(label=label, command=argv, returncode=result.returncode,
            stdout=result.stdout, stderr=result.stderr, utc=datetime.now(timezone.utc).isoformat())
        self.state['attempts'].append(record)
        self.update()
        if result.returncode:
            raise RuntimeError(f'{label} failed; preserve submitted work and inspect the saved output')
        return result.stdout

    def metadata(self, paths, kind='full'):
        """Read only small result/descriptor metadata over the existing transport."""
        sys.path.insert(0, str(SCHEDULER.parent))
        import scheduler
        source = ("import json; from pathlib import Path; root=Path("+repr(REMOTE)+
            "); paths="+repr(paths)+"; result={}\n"
            "for name in paths:\n"
            " data=json.loads((root/name).read_text())\n")
        if kind == 'descriptor':
            source += " result[name]=dict(stage=data['stage'],manifest=data['manifest'],jobs=len(data['jobs']))\n"
        elif kind == 'stage':
            source += (" result[name]={k:data.get(k) for k in ['status','stage','jobs','truth_used']}\n"
                " result[name]['failed_jobs']=[{k:r.get(k) for k in ['label','seed','arm','returncode']} for r in data.get('failed_jobs',[])]\n")
        elif kind == 'seal':
            source += " result[name]={k:data.get(k) for k in ['status','registered_pairs','registered_terminal_cells','completed_old','completed_new','truth_access_allowed','truth_used']}\n"
        else:
            source += " result[name]=data\n"
        source += 'print(json.dumps(result))\n'
        rc, stdout, stderr = scheduler.run_on('node002', shlex.join([PYTHON, '-c', source]),
            timeout=45, check=False)
        self.state['attempts'].append(dict(label='metadata/'+kind, paths=paths,
            returncode=rc, stderr=stderr, payload_bytes=len(stdout.encode()),
            utc=datetime.now(timezone.utc).isoformat()))
        self.update()
        if rc:
            raise RuntimeError('Small metadata read failed; no scientific task is resubmitted')
        return json.loads(stdout)

    def task_states(self, ids):
        tasks = read(Path.home()/'.claude/scheduler/queue.json')['tasks']
        rows = tasks.values() if isinstance(tasks, dict) else tasks
        selected = {t['id']:t for t in rows if t['id'] in ids}
        if set(selected) != set(ids):
            raise RuntimeError('Submitted task missing from live scheduler metadata')
        return [{k:selected[i].get(k) for k in ['id','status','node','cpu_cores','est_vram_mb',
            'exit_code','started_at','finished_at','last_block_reason','log_path']} for i in ids]

    def wait(self, ids, label):
        previous, last_print = None, 0.
        while True:
            rows = self.task_states(ids)
            self.update(waiting_for=label, latest_tasks=rows)
            counts = {s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})}
            if counts != previous or time.monotonic()-last_print >= 60:
                print(json.dumps(dict(stage=label, scheduler_status=counts)), flush=True)
                previous, last_print = counts, time.monotonic()
            if any(r['status'] in ('failed','cancelled','lost') or
                   (r['status']=='done' and r['exit_code'] not in (None,0)) for r in rows):
                raise RuntimeError(f'Scheduler infrastructure failure in {label}; no scientific retry')
            if all(r['status']=='done' for r in rows):
                return rows
            time.sleep(self.poll_seconds)

    def job(self, label, items, command):
        entry = self.state['jobs'].setdefault(label, {})
        if entry.get('status') == 'complete':
            return entry
        if not items:
            entry.update(status='complete', task_ids=[], records=[])
            self.update()
            return entry
        if 'task_ids' not in entry:
            if entry.get('submission_started'):
                raise RuntimeError(f'{label}: interrupted submission; reconcile saved scheduler IDs before resuming')
            entry['submission_started'] = datetime.now(timezone.utc).isoformat()
            record_template = str(Path(self.plan['output_dir'])/'cluster_control/jobs'/label/'{start}_{end}.json')
            shell = shlex.join(['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',
                'PYTHONPATH=.','PYTHONFAULTHANDLER=1','PYTEST_DISABLE_PLUGIN_AUTOLOAD=1',
                *module('performance.run_mainline_safe_unsampled_cluster_job','--record',record_template,'--',*command)])
            argv = [sys.executable,str(SCHEDULER),'submit-cpu-batch','--items',str(items),
                '--nodes',','.join(NODES),'--cmd-template',shell,'--cwd',REMOTE,
                '--signature',f"KG_op/{self.plan['name']}/{self.signature_stage}/{label}/{{start}}-{{end}}",
                '--description',f'SC-OLH-KG {self.stage_label} {label} {{node}} [{{start}},{{end}})',
                '--ram-mb','32768','--project','KG_op','--skip-launch-staging','--env-spec','none','--json']
            self.update(status='running', active_stage=label)
            stdout = self.capture(label+'/submit', argv)
            ids = re.findall(r'^submitted (t\d+)\b', stdout, re.MULTILINE)
            if not ids:
                raise RuntimeError('No scheduler IDs in submission output; inspect before any retry')
            payload = stdout[:stdout.index('\nsubmitted ')]
            plan = json.loads(payload)
            entry.update(task_ids=ids, plan=plan,
                record_paths=[record_template.format(**r) for r in plan['plan']])
            self.update()
        entry['scheduler_tasks'] = self.wait(entry['task_ids'], label)
        entry['records'] = self.metadata(entry['record_paths'])
        bad = [r for r in entry['records'].values() if r.get('status') != 'complete' or r.get('child_returncode') != 0]
        self.update()
        if bad:
            raise RuntimeError(f'{label}: retained child failure or incomplete launch; no automatic resubmission')
        entry['status'] = 'complete'
        self.update()
        return entry

    def run_stage(self, stage):
        state = self.state['stages'].setdefault(stage,{})
        if state.get('status') == 'complete':
            return
        index = STAGES.index(stage)
        if index and self.state['stages'].get(STAGES[index-1],{}).get('status') != 'complete':
            raise RuntimeError('The preceding global stage is not complete')
        descriptor = str(Path(self.plan['output_dir'])/'cluster'/stage/'jobs.json')
        self.job(stage+'/describe',1,module('performance.mainline_cluster_dispatch','describe',
            '--manifest',self.manifest,'--stage',stage))
        detail = self.metadata([descriptor],'descriptor')[descriptor]
        state['registered_jobs'] = detail['jobs']
        self.job(stage+'/workers',detail['jobs'],module('performance.mainline_cluster_dispatch','worker',
            '--descriptor',descriptor,'--start','{start}','--end','{end}','--workers','{workers}'))
        self.job(stage+'/finish',1,module('performance.mainline_cluster_dispatch','finish','--descriptor',descriptor))
        completion = str(Path(self.plan['output_dir'])/'cluster'/stage/'completion.json')
        result = self.metadata([completion],'stage')[completion]
        if result['status'] != 'complete' or result['truth_used']:
            raise RuntimeError('Stage completion did not satisfy the frozen pre-truth barrier')
        state.update(status='complete',completion=result)
        self.update()
        print(json.dumps(dict(stage=stage,status='complete',jobs=result['jobs'],
            failed_jobs=len(result['failed_jobs']))),flush=True)

    def run(self):
        if self.state['status'] == 'complete':
            print('Stage73 controller already complete',flush=True)
            return
        try:
            for stage in STAGES:
                self.run_stage(stage)
            seal_path = str(Path(self.plan['output_dir'])/'execution_status.json')
            seal = self.metadata([seal_path],'seal')[seal_path]
            if seal['status'] != 'sealed' or seal['registered_terminal_cells'] != 432 or seal['truth_used']:
                raise RuntimeError('All 432 terminal outcomes must be sealed before inventory and truth')
            self.state['terminal_seal'] = seal
            self.update()
            self.job('audit/freeze',1,module(AUDIT,'freeze','--manifest',self.manifest))
            self.job('audit/tasks',24,module('performance.run_mainline_safe_unsampled_audit_batch',
                '--manifest',self.manifest,'--start','{start}','--end','{end}','--workers','{workers}'))
            self.job('audit/collect',1,module(AUDIT,'collect','--manifest',self.manifest))
            self.update(status='complete',waiting_for=None)
        except Exception as error:
            self.update(status='blocked',error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default=MANIFEST)
    parser.add_argument('--poll-seconds',type=int,default=30)
    args = parser.parse_args()
    Controller(args.manifest,args.poll_seconds).run()


if __name__ == '__main__':
    main()
