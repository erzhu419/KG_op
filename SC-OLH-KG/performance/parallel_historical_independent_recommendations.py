"""Freeze independent model shards, then assemble the full verification barrier."""

import argparse
import json
import shlex
import shutil

from performance.complete_historical_source_heads import ROOT, read, save
from performance.prepare_historical_independent_query_parallel import LOCAL, REMOTE, PYTHON, NODES
from performance.run_historical_independent_kg_pilot import PLAN, inputs, key, validate_recommendations

OUTPUT = 'paper_artifacts/historical_independent_recommendations_parallel_v1'


def prepare():
    spec = read(PLAN); pilot, _, states, _ = inputs(spec)
    out = ROOT/OUTPUT; out.mkdir(parents=True, exist_ok=True)
    plans = ROOT/'performance/manifests/historical_independent_recommendations_parallel_v1'; plans.mkdir(exist_ok=True)
    specs = []; shards = []; files = {'performance/run_historical_independent_kg_pilot.py',
        'performance/parallel_historical_independent_recommendations.py'}
    baselines = read(spec['baselines']); files.update(c['result'] for c in baselines['cases'])
    connection = read(pilot['source_connection_plan']); files.add(read(connection['legacy_connection_plan'])['public_inputs'])
    for state in states:
        plan = dict(spec, state_names=[state['name']], output=OUTPUT+'/shards/'+state['name'],
            scope='Resource-only per-model parallel freeze. Require the same complete144-query registration. '
                'Preserve frozen queries, raw search streams, forty actions and numerical gates. '
                'A shard never creates the global48-recommendation verification barrier.')
        path = plans/(state['name']+'.json'); save(path, plan); files.add(str(path.relative_to(ROOT)))
        cmd = ['env', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'PYTHONPATH=.',
            PYTHON, '-u', '-m', 'performance.run_historical_independent_kg_pilot', 'freeze',
            '--plan', str(path.relative_to(ROOT))]
        specs.append({'description': 'Independent recommendation '+state['name'], 'cmd': shlex.join(cmd),
            'cwd': str(LOCAL), 'signature': 'KG_op/historical_independent_recommendations_parallel_v1/'+state['name'],
            'project': 'KG-SYNTH', 'vram': 0, 'cpu': 1, 'ram_mb': 4096,
            'allowed_nodes': NODES, 'skip_launch_staging': True})
        shards.append({'state_name': state['name'], 'plan': str(path.relative_to(ROOT)), 'output': plan['output']})
    save(out/'deployment.json', {'execution_plan': PLAN, 'shards': shards, 'global_output': spec['output']})
    files.add(OUTPUT+'/deployment.json')
    (out/'scheduler_specs.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in specs))
    (out/'upload_files.txt').write_text('\n'.join(sorted(files))+'\n')
    for f in files:
        if f.endswith('.py') or f.startswith('performance/manifests/'):
            target = LOCAL/f; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/f, target)
    print(json.dumps({'shards': len(shards), 'upload_bytes': sum((ROOT/f).stat().st_size for f in files)}), flush=True)


def collect():
    deployment = read(OUTPUT+'/deployment.json'); spec = read(deployment['execution_plan'])
    pilot, _, states, _ = inputs(spec); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    rows = []; cpu = 0.; decisions = None
    for shard in deployment['shards']:
        directory = ROOT/shard['output']; progress = read(str(directory/'recommendation_progress.json'))
        state = next(s for s in states if s['name'] == shard['state_name'])
        if not progress['selected_recommendations_complete'] or progress['independent_outcomes_opened']:
            raise ValueError('Every recommendation shard must complete before assembly')
        validate_recommendations(progress['rows'], [state], pilot)
        rows.extend(progress['rows']); cpu += progress['cpu_seconds']
        current = read(str(directory/'query_decisions.json'))
        if decisions is not None and current['rows'] != decisions['rows']:
            raise ValueError('Shards used different frozen query decisions')
        decisions = current
        # Move the one actual query record with its numerical recommendation, on the server.
        source = directory/state['name']; target = out/state['name']
        if source.exists():
            target.mkdir(exist_ok=True)
            for path in source.glob('*.json'):
                destination = target/path.name
                if destination.exists():
                    if read(str(destination)) != read(str(path)):
                        raise ValueError('Existing global query record differs from its shard')
                else:
                    shutil.copyfile(path, destination)
    validate_recommendations(rows, states, pilot)
    indexed = {key(r): r for r in rows}
    rows = [indexed[(s['task_id'], s['layout'], s['arm'], p)] for s in states for p in pilot['policies']]
    result = {'execution_plan': deployment['execution_plan'], 'rows': rows, 'cpu_seconds': cpu,
        'all_recommendations_frozen': True, 'selected_recommendations_complete': True,
        'independent_outcomes_opened': False, 'independent_confirmation': False}
    save(out/'query_decisions.json', decisions)
    save(out/'recommendation_progress.json', result); save(out/'recommendations.json', result)
    print(json.dumps({'recommendations_frozen':len(rows), 'physical_search_calls':
        sum(read(str(p))['physical_calls'] for p in out.glob('*/search_observation.json'))}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'collect'])
    args = parser.parse_args(); (prepare if args.stage == 'prepare' else collect)()
