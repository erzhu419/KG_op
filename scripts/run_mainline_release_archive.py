import json, resource, subprocess, sys, tarfile, time
from pathlib import Path
root, publication, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
output.mkdir(parents=True, exist_ok=False)
archive = output / 'current_mainline_review_20261026.tar.gz'
release = root / 'SC-OLH-KG/reproducibility/mainline_release'
started, wall = time.process_time(), time.perf_counter()
before = resource.getrusage(resource.RUSAGE_CHILDREN)
receipt = dict(schema='mainline-final-archive-assembly-v1', status='failed', exit_code=1,
    publication_commit=publication, automatic_retries=0, scientific_calls=0,
    model_restorations=0, historical_reserves_reset=False, phases=[])
try:
    phases = [('frozen_behavior_parity',[sys.executable,str(release/'frozen_behavior_parity.py'),
            '--old-case',str(release/'case/reference/search'),'--old-input',str(release/'case/reference/input.json'),
            '--new-output',str(root/'SC-OLH-KG/paper_artifacts/mainline_submission_release_v1/clean_small_attempt_5/small_attempt'),
            '--case-json',str(release/'case/case.json'),'--output',str(release/'receipts/frozen_behavior_parity.json')]),
            ('build',[sys.executable,str(release/'package.py'),'build','--root',str(root),
            '--release-commit',publication,'--output',str(archive)]),
            ('verify',[sys.executable,str(release/'package.py'),'verify','--output',str(archive)])]
    parity = release / 'receipts/frozen_behavior_parity.json'
    if parity.exists():
        previous = json.loads(parity.read_text())
        assert previous['status'] == 'passed' and previous['exit_code'] == 0
        receipt['phases'].append(dict(name='reuse_passed_frozen_behavior_parity',exit_code=0))
        phases = phases[1:]
    for name, args in phases:
        with (output/(name+'.log')).open('w') as log:
            result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, cwd=root)
        receipt['phases'].append(dict(name=name,exit_code=result.returncode))
        if result.returncode:
            raise RuntimeError(name+' failed; its log and archive are retained')
    with tarfile.open(archive,'r:gz') as source:
        manifest = json.load(source.extractfile('release_manifest.json'))
    (output/'release_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if not manifest['delivery_ready']:
        raise RuntimeError('Final archive has not passed its recorded acceptance gates')
    receipt.update(status='complete',exit_code=0,archive_name=archive.name,archive_bytes=archive.stat().st_size,
        payload_files=len(manifest['files']),acceptance_receipts=manifest['acceptance_receipts'],
        delivery_ready=True)
except BaseException as error:
    receipt['error']=repr(error)
    raise
finally:
    child=resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt.update(operator_cpu_seconds=time.process_time()-started,
        child_cpu_seconds=child.ru_utime+child.ru_stime-before.ru_utime-before.ru_stime,
        wall_seconds=time.perf_counter()-wall,child_peak_rss_bytes=int(child.ru_maxrss)*1024)
    (output/'assembly_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
