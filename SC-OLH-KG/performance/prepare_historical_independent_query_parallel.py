"""Package cached independent queries and merge isolated scheduler shards."""

import argparse
import copy
import gzip
import json
from pathlib import Path
import shlex
import time
from unittest.mock import patch

from core.raw_terminal_head_grid import evaluate_head_records
from performance.check_historical_raw_head_grid import terminal_for
from performance.complete_historical_source_heads import ROOT, read, save
from performance.prepare_historical_independent_kg import restore_target_models
from performance.refine_historical_independent_query_axes import PLAN, compact, merge_cases
from performance.check_historical_independent_queries import cohort_status

OUTPUT = 'paper_artifacts/historical_independent_query_parallel_v1'
LOCAL = ROOT.parent.parent/'KG_op_independent_kg_parallel_deploy'/'SC-OLH-KG'
REMOTE = '/home/zhengliang01/scheduleurm_work/KG_op_independent_kg_parallel_deploy/SC-OLH-KG'
PYTHON = '/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python'
NODES = ['node001', 'node002', 'node003', 'node004', 'node005', 'node006']


def prepare():
    out = ROOT/OUTPUT; out.mkdir(parents=True, exist_ok=True)
    files = set(); original_read = Path.read_text
    def tracked_read(path, *args, **kwargs):
        if path.is_relative_to(ROOT):
            files.add(str(path.relative_to(ROOT)))
        return original_read(path, *args, **kwargs)
    with patch.object(Path, 'read_text', tracked_read):
        spec = read(PLAN); parent = read(spec['parent_plan']); pilot = read(parent['pilot_plan'])
        original = read(parent['output']+'/results.json'); states = read(parent['preflight'])['states']
        read(pilot['source_states']); read(parent['preflight_output']+'/frozen_inputs.json'); read(parent['baselines'])
        selected = []
        for state in states:
            if not any((s['layout'], s['arm']) == (state['layout'], state['arm']) for s in selected):
                selected.append(state)
        smoke = []
        for state, model, _ in restore_target_models(pilot, selected):
            if smoke or state['arm'] != 'source_structural_mixture':
                continue
            case = next(c for c in original['cases'] if c['state_name'] == state['name'])
            bank = read(case['bank']); h = bank['heads'][1]
            positions = h['positions'][:4]; labels = dict(zip(h['positions'], h['labels']))
            args = (model, terminal_for(state), 360, case['query'], 1, labels, positions)
            before_weights = model.weights.copy(); history = copy.deepcopy(model.history)
            outputs = []
            for workers in (1, 4):
                tick = time.monotonic()
                rows = evaluate_head_records(*args, powers=[12], seed=read(pilot['numerical_plan'])['seed'],
                    utility_identities=[4, 5], guarded_mean_position=0, workers=workers)
                outputs.append([{k: v for k, v in r.items() if k != 'cpu_seconds'} for r in rows])
                smoke.append({'workers': workers, 'wall_seconds': time.monotonic()-tick})
            assert outputs[0] == outputs[1]
            assert (model.weights == before_weights).all() and model.history == history
    pending = [c for c in original['cases'] if not c.get('adaptation_diagnostics_passed', False)]
    merge_cases(states, original['cases'], [])
    specs = []; shards = []
    plans = ROOT/'performance/manifests/historical_independent_query_parallel_v1'; plans.mkdir(exist_ok=True)
    for case in pending:
        heavy = not case['stages'][-1]['grid_diagnostics_passed']
        workers = 16 if heavy else (2 if case['arm'] == 'source_structural_mixture' else 1)
        plan = dict(spec, status='Resource-only per-query parallel continuation; original scientific registration unchanged.',
            case_names=[case['name']], workers=workers, maximum_single_thread_cpu_seconds=7200 if heavy else 900,
            output=OUTPUT+'/shards/'+case['name'],
            scope='Reuse every original head and partial power; independently complete this query under unchanged gates. '
                'All outputs are isolated by query. No new source fits, paid target episodes or deployment outcomes.')
        path = plans/(case['name']+'.json'); save(path, plan); files.add(str(path.relative_to(ROOT)))
        command = ['env', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
            'PYTHONPATH=.', PYTHON, '-u', '-m', 'performance.refine_historical_independent_query_axes',
            '--plan', str(path.relative_to(ROOT))]
        specs.append({'description': 'Independent KG query '+case['name'], 'cmd': shlex.join(command),
            'cwd': str(LOCAL), 'signature': 'KG_op/historical_independent_query_parallel_v1/'+case['name'],
            'project': 'KG-SYNTH', 'vram': 0, 'cpu': workers, 'ram_mb': 8192 if heavy else 3072,
            'allowed_nodes': NODES, 'skip_launch_staging': True})
        shards.append({'name': case['name'], 'plan': str(path.relative_to(ROOT)), 'output': plan['output'],
            'workers': workers, 'risk_refinement': heavy})
    # These cached files supply the original heads, predictive banks and partial refinements.
    files.update(str(p.relative_to(ROOT)) for p in (ROOT/parent['output']).rglob('*.json'))
    files.update(c['bank'] for c in original['cases'])
    # Python packages are small; archives, profiles and unrelated result trees are excluded.
    for name in ('performance', 'core', 'representation', 'acquisition', 'baselines', 'problems', 'utils',
                 'algorithms', 'encoders', 'variance', 'data'):
        files.update(str(p.relative_to(ROOT)) for p in (ROOT/name).rglob('*.py'))
    files.update(str(p.relative_to(ROOT)) for p in (ROOT/'performance/manifests').glob('*.json'))
    deployment = {'local_code_root': str(LOCAL), 'remote_code_root': REMOTE, 'nodes': NODES,
        'original_plan': spec['parent_plan'], 'global_output': spec['output'], 'shards': shards,
        'source_prior_restorations_checked': len(selected), 'native_parallel_smoke': smoke,
        'native_parallel_results_identical': True, 'new_physical_calls': 0, 'new_verifier_calls': 0,
        'shard_count': len(shards), 'declared_cpu_sum': sum(s['cpu'] for s in specs)}
    save(out/'deployment.json', deployment); files.add(OUTPUT+'/deployment.json')
    (out/'scheduler_specs.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in specs))
    (out/'upload_files.txt').write_text('\n'.join(sorted(files))+'\n')
    print(json.dumps({k: deployment[k] for k in ('shard_count', 'declared_cpu_sum', 'native_parallel_smoke')}
        | {'upload_files': len(files), 'upload_bytes': sum((ROOT/f).stat().st_size for f in files)}), flush=True)
    return deployment


def collect():
    deployment = read(OUTPUT+'/deployment.json'); parent = read(deployment['original_plan'])
    original = read(parent['output']+'/results.json'); states = read(parent['preflight'])['states']
    cases = []; incomplete = []; cpu = 0.
    for shard in deployment['shards']:
        path = ROOT/shard['output']/'results.json'
        if not path.exists():
            incomplete.append(shard['name']); continue
        result = read(str(path)); cpu += result['cpu_seconds']
        if len(result['cases']) != 1 or result['cases'][0]['name'] != shard['name']:
            incomplete.append(shard['name']); continue
        cases.extend(result['cases'])
        if not cases[-1].get('processing_complete', False):
            incomplete.append(shard['name'])
    merged = merge_cases(states, original['cases'], cases)
    result = dict(cohort_status(states, merged), execution_plan=OUTPUT+'/deployment.json',
        cases=[compact(c) for c in merged], pending_shards=incomplete, cpu_seconds=cpu,
        all_shards_processed=not incomplete, independent_outcomes_opened=False,
        cost={'new_source_fits': 0, 'new_physical_calls': 0, 'new_verifier_calls': 0})
    out = ROOT/deployment['global_output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'phase_results.json', result)
    save(ROOT/OUTPUT/'phase_results.json', result)
    (ROOT/OUTPUT/'phase_results.json.gz').write_bytes(gzip.compress(json.dumps(result).encode()))
    print(json.dumps({k: result[k] for k in ('registered_query_cases', 'passed_query_cases',
        'all_model_states_ready', 'all_shards_processed')} | {'pending_shards': len(incomplete)}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'collect'])
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else collect)()
