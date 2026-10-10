"""Task-deletion attribution of risk-law and learned-map instability."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_diverse_source_design import encode_diverse_source
from performance.check_legacy_public_mean_source_half import fit_risk
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_joint_population_independent_pilot import target_problem
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import source_prior_moments


def read(path):
    return json.loads((ROOT/path).read_text())


def query_rows(mode, deleted, inputs, fits, queries):
    rows = []; basis = OrthogonalLowOrderBasis(inputs['public_risk_reference'])
    for h, name in zip(fits['heads'], ('eta_g', 'eta_f')):
        prior = restore_joint(basis, h['joint_structural'])
        for query, psi in queries.items():
            mean, covariance = source_prior_moments(prior, np.asarray(psi))
            rows.append({'mode': mode, 'deleted_task': deleted, 'head': name, 'query': query,
                         'mean': mean.tolist(), 'variance': covariance.diagonal().tolist()})
    return rows


def run():
    spec = read('performance/manifests/legacy_joint_population_public_mean_risk_stability_execution_v1.json')
    source = read(spec['source_execution']); base = read(spec['source_inputs'])
    base_fit = read(spec['reference_risk_fit']); target_inputs = read(spec['reference_target_inputs'])
    data = read(spec['source_data']); public = read(Path(spec['source_data']).with_name('frozen_inputs.json'))
    bank = [tuple(p) for p in public['bank']]; target_bank = [tuple(p) for p in target_inputs['bank']]
    descriptors = {i: legacy_exposure_descriptors(no_outcomes(problem_for(public['plan'], t)), bank, 50)
                   for i, t in enumerate(public['tasks'])}
    target_desc = {t['task_id']: legacy_exposure_descriptors(
        no_outcomes(target_problem(target_inputs['target_plan'], t['config'])), target_bank, 50)
        for t in target_inputs['targets']}
    base_queries = {'source_reference': base['public_risk_reference'],
        **{f'source_holdout_{i}': base['heldout_risk_psi'][str(i)] for i in source['heldout_tasks']},
        **{t['task_id']: t['risk_psi'] for t in target_inputs['targets']}}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    rows = query_rows('full144_reference', None, base, base_fit, base_queries)
    started = time.process_time(); fitting_cpu = 0.; continuations = 0
    ids = source['training_tasks']; layouts = source['source_site_indices_by_task']
    for mode in spec['modes']:
        for deleted in spec['deletion_tasks']:
            keep = [(i, sites) for i, sites in zip(ids, layouts) if i != deleted]
            branch = out/mode/f'task{deleted}'; branch.mkdir(parents=True, exist_ok=True)
            if mode == 'risk_fit_only_at_frozen48_label_map':
                inputs = {**base, 'risk_source_tasks': [t for i, t in zip(ids, base['risk_source_tasks']) if i != deleted]}
                queries = base_queries
            else:
                coordinate, tasks, center, scale = encode_diverse_source(data,
                    [i for i, _ in keep], [s for _, s in keep], descriptors, public['plan']['task_distribution']['L'])
                # Keep the same response-free public catalog provider for both modes.
                reference = coordinate.transform(descriptors[ids[0]])
                inputs = {'risk_source_tasks': [{**t, **{k: t[k].tolist()
                    for k in ('psi', 'features', 'response', 'noise')}} for t in tasks],
                    'public_risk_reference': reference.tolist(), 'coordinate': coordinate.diagnostics(),
                    'risk_coordinate_center': center.tolist(), 'risk_coordinate_scale': scale.tolist()}
                queries = {'source_reference': reference.tolist(),
                    **{f'source_holdout_{i}': coordinate.transform(descriptors[i]).tolist() for i in source['heldout_tasks']},
                    **{name: coordinate.transform(desc).tolist() for name, desc in target_desc.items()}}
            save(branch/'inputs.json', inputs)
            fits = fit_risk({**spec, 'output': str(branch.relative_to(ROOT))}, inputs)
            fitting_cpu += fits['cpu_seconds']
            continuations += sum('continued_from_optimizer_iterations' in h['joint_structural'] for h in fits['heads'])
            rows.extend(query_rows(mode, deleted, inputs, fits, queries))
            print('risk deletion forecasts', mode, deleted, flush=True)
    save(out/'predictions.json', {'execution': spec, 'rows': rows})
    # No outcomes have been read by the branch fits or query forecasts.
    scoring = read(spec['target_scoring']); summary = {}
    for mode in spec['modes']:
        summary[mode] = {}
        for head, truth_column in (('eta_g', 3), ('eta_f', 1)):
            points = {}
            for j in spec['diagnostic_candidates']:
                values = []
                for target in target_inputs['targets']:
                    name = target['task_id']
                    ref = next(r for r in rows if r['mode'] == 'full144_reference' and r['head'] == head and r['query'] == name)
                    deletion = [r for r in rows if r['mode'] == mode and r['head'] == head and r['query'] == name]
                    means = np.array([r['mean'][j] for r in deletion])
                    truth = next(f for f in scoring['folds'] if f['task_id'] == name)['scoring_truth'][j][truth_column]
                    values.append({'task_id': name, 'full_prior_bias': ref['mean'][j]-truth,
                        'full_prior_sd': float(np.sqrt(ref['variance'][j])),
                        'deletion_prior_mean_sd': float(means.std(ddof=1)),
                        'deletion_prior_mean_range': float(means.max()-means.min()),
                        'deletion_mean_minus_full': float(means.mean()-ref['mean'][j])})
                points[str(j)] = {'tasks': values, **{k: float(np.mean([v[k] for v in values])) for k in
                    ('full_prior_bias', 'full_prior_sd', 'deletion_prior_mean_sd',
                     'deletion_prior_mean_range', 'deletion_mean_minus_full')}}
            summary[mode][head] = points
    contrasts = []
    fixed, relearned = spec['modes']
    for head in ('eta_g', 'eta_f'):
        for query in base_queries:
            delta = np.array([np.asarray(next(r for r in rows if r['mode'] == relearned
                and r['deleted_task'] == i and r['head'] == head and r['query'] == query)['mean'])
                -np.asarray(next(r for r in rows if r['mode'] == fixed and r['deleted_task'] == i
                and r['head'] == head and r['query'] == query)['mean']) for i in spec['deletion_tasks']])
            contrasts.append({'head': head, 'query': query, 'paired_map_refit_delta_rms': float(np.sqrt((delta**2).mean())),
                'points': {str(j): {'mean_delta': float(delta[:, j].mean()),
                    'rms_delta': float(np.sqrt((delta[:, j]**2).mean()))} for j in spec['diagnostic_candidates']}})
    result = {'execution': spec, 'summary': summary, 'paired_refit_contrasts': contrasts,
        'cost': {'historical_source_calls_total': 144, 'source_records_per_deleted_fit': 40,
            'new_native_source_head_fits': 72+continuations, 'new_simulator_calls': 0,
            'new_verification_calls': 0, 'new_terminal_integrations': 0},
        'source_fit_cpu_seconds': fitting_cpu, 'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)
    print(json.dumps({'points63': {m: {h: {k: v for k, v in summary[m][h]['63'].items() if k != 'tasks'}
        for h in ('eta_g', 'eta_f')} for m in spec['modes']},
        'cpu_seconds': result['cpu_seconds'], 'fit_cpu_seconds': fitting_cpu}), flush=True)


if __name__ == '__main__':
    run()
