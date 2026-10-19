#!/usr/bin/env python3
"""Wait for the already-submitted validation once, then continue its frozen cohort."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import sys

import run_mainline_terminal_recording_cluster as recording


def main():
    control = recording.Controller()
    control.output.mkdir(parents=True, exist_ok=True)
    lock = (control.output / 'continuation.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()) + '\n')
    lock.flush()
    try:
        control.wait(['t169548'], 'validation/attempt1')
        sys.path.insert(0, str(recording.base.SCHEDULER.parent))
        import scheduler
        source = '''import json,re,time
from pathlib import Path
root=Path(REMOTE)
output=root/'paper_artifacts/mainline_terminal_recording_seed_replication_v1'
names=['cluster_control/jobs/validation/attempt1_0_1.json',
       'cluster_control/validation/attempt1/checks.json',
       'cluster_control/validation/attempt1/admission.json']
values={name:json.loads((output/name).read_text()) for name in names}
job,checks,admission=[values[name] for name in names]
log=(root/job['log_file']).read_text()
assert job['status']=='complete' and job['child_returncode']==0, 'Preserve failed validation wrapper'
assert checks['status']=='complete' and checks['returncode']==0 and re.search(r'\\b18 passed\\b',log), '18 checks did not pass'
assert admission['status']=='validated' and not admission['execution_started']
history_file=output/'cluster_control/validation/attempt1/history_accounts.json'
if history_file.exists():
 history=json.loads(history_file.read_text())
else:
 started=time.process_time()
 ref='paper_artifacts/mainline_terminal_recording_preparation_v1/scheduler_execution.json'
 paths=[root/path for path in admission['manifests']]
 changed=[]
 for path in paths:
  plan=json.loads(path.read_text())
  accounts=plan.setdefault('historical_accounts',[])
  if ref not in accounts:
   accounts.append(ref);path.write_text(json.dumps(plan,indent=2)+'\\n');changed.append(str(path.relative_to(root)))
 history=dict(status='complete',changed_manifests=changed,historical_account=ref,
              cpu_seconds=time.process_time()-started,new_scientific_calls=0)
 history_file.write_text(json.dumps(history,indent=2)+'\\n')
values['cluster_control/validation/attempt1/history_accounts.json']=history
print(json.dumps(dict(records=values,log=log,log_file=job['log_file'])))
'''.replace('REMOTE', repr(recording.base.REMOTE))
        rc, stdout, stderr = scheduler.run_on('node002',
            shlex.join([recording.base.PYTHON, '-c', source]), timeout=45, check=False)
        control.state['attempts'].append(dict(label='validation/admission_gate',
            returncode=rc, stderr=stderr, payload_bytes=len(stdout.encode()),
            utc=datetime.now(timezone.utc).isoformat()))
        control.update()
        if rc:
            raise RuntimeError('Validation or metadata admission failed; no re-test or cohort submission')
        value = json.loads(stdout)
        for name, data in value['records'].items():
            recording.base.save(recording.base.PROJECT / control.plan['output_dir'] / name, data)
        log = recording.base.PROJECT / value['log_file']
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(value['log'])
        control.update(validation=dict(status='complete', passed=18,
            task_id='t169548', records=value['records']))
        print(json.dumps(dict(validation='18 passed', admission='validated',
                              controller_pid=os.getpid())), flush=True)
        control.run()
    except Exception as error:
        control.update(status='blocked', error=str(error))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
