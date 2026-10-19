"""Connect frozen source noise geometry to existing d1000 decision records."""
from datetime import datetime, timezone
import json

import numpy as np

from baselines.transfer_archive import FrozenTransferArchive
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.mainline_target_noise_evidence import evidence, distribution
from performance.mainline_target_noise_geometry import prepare_source_geometry, target_geometry
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save, load_pickle

MANIFEST = ROOT/'performance/manifests/mainline_target_noise_geometry_v1.json'


def describe_geometry(row, ids):
    ids = np.asarray(ids, dtype=int)
    if not len(ids):
        return dict(policy_count=0)
    return dict(policy_count=len(ids),
        encoder_outside_source_count=int(row['encoder_outside_source'][ids].sum()),
        encoder_maximum_standardized_range_excess=float(
            row['standardized_encoder_range_excess'][ids].max()),
        eta_maps=[dict(kind=m['kind'], support=m['support'], heads=m['heads'],
            source_identity_mass=m['source_identity_mass'],
            norm_ratio=distribution(m['design_norm_to_source_max'][ids]),
            count_above_source_max=int(np.sum(m['design_norm_to_source_max'][ids] > 1.+1e-10)))
            for m in row['eta_maps']],
        source0={head: {key: distribution(values[key][ids]) for key in
                       ('mean_eta','variance_eta','noise')}
                 for head,values in row['source0'].items()})


def point_geometry(row, index):
    return dict(full_bank_id=index, encoder_values=row['encoder_values'][index],
        encoder_outside_source=bool(row['encoder_outside_source'][index]),
        descriptor_second_bin=float(row['descriptor_second_bin'][index]),
        eta_maps=[dict(kind=m['kind'], support=m['support'], heads=m['heads'],
            source_identity_mass=m['source_identity_mass'],
            design_norm=float(m['design_norm'][index]),
            training_design_norm_max=m['training_design_norm_max'],
            norm_ratio=float(m['design_norm_to_source_max'][index])) for m in row['eta_maps']],
        source0={head:dict(**{key:float(values[key][index]) for key in
                              ('mean_eta','variance_eta','noise')},
                    variance_parts={key:float(value[index]) for key,value in values['variance_parts'].items()})
                 for head,values in row['source0'].items()})


def main():
    plan = read(MANIFEST); output = ROOT/plan['output_dir']
    if (output/'completion.json').exists():
        print('reuse completed diagnostic', flush=True)
        return
    with attempt(output, 'frozen-source-geometry-and-saved-decision-evidence') as cost:
        cost.update(new_source_fits=0, new_target_initial_calls=0,
            new_truth_function_calls=0, saved_raw_rows_replayed=0, download_bytes=0,
            source_model_restorations=0, saved_engine_reads=0)
        saved = evidence(plan)
        save(output/'evidence.json', saved)
        bundle = load_pickle(ROOT/plan['source_models_file'])
        cost['source_model_restorations'] += 1
        archive = FrozenTransferArchive.load(ROOT/plan['source_archive'])
        providers = {task.name:no_outcomes(build_scalarized_problem(
            task.name, task.X.shape[1], 100, .04, .05, [.5,.5])) for task in archive.tasks}
        source = prepare_source_geometry(bundle, archive, providers)
        source_summary = dict(source['summary'],
            eta_maps=[dict(key=m['key'], source_identity_mass=m['source_identity_mass'],
                selected_features=m['selected_features'],
                training_design_norm=distribution(m['training_design_norm'])) for m in source['eta_maps']],
            source0={head:{key:distribution(values[key]) for key in ('mean_eta','variance_eta','noise')}
                     for head,values in source['source0'].items()},
            source_identity_weights=dict(zip(bundle['identities'], np.exp(bundle['log_weights']))))
        save(output/'source_geometry.json', source_summary)
        summaries, by_point = [], {}
        for seed in plan['target_seeds']:
            directory=ROOT/plan['source_output_dir']/f'seed_{seed}'/'legacy_role_full'
            with np.load(directory/'query_policies.npz') as data:
                points = data['points']
            engine = load_pickle(directory/'search_engine.pkl')
            cost['saved_engine_reads'] += 1
            row = target_geometry(bundle, source, no_outcomes(engine.problem), points)
            groups=saved['geometry_groups_by_seed'][str(seed)]
            focus=sorted(set(groups['first_divergence_selected_pair']+groups['focal_missed_good']+
                             groups['initial']+groups['observed_queries']))
            point_rows=[point_geometry(row, index) for index in focus]
            actual={tuple(r['point']):r['truth'] for r in read(directory.parent/'truth.json')}
            for record in point_rows:
                truth=actual.get(tuple(points[record['full_bank_id']]))
                if truth is not None:
                    record['source0_risk_sigma_ratio']=float(
                        np.sqrt(record['source0']['g']['noise'])/truth['constraint_sigma'])
                by_point[seed,record['full_bank_id']]=record
            result=dict(seed=seed, bank_summary=row['summary'],
                groups={name:describe_geometry(row, ids) for name,ids in groups.items()},
                focused_policies=point_rows)
            save(output/f'seed_{seed}.json', result)
            summaries.append(dict(seed=seed, **result['groups']))
            print('geometry complete',seed,'bank',len(points),'scored',len(groups['all_scored']),flush=True)
        first=[]
        for case in saved['first_divergences']:
            first.append(dict(case, geometry=[by_point[case['seed'],r['full_bank_id']]
                                               for r in case['selected_pair']]))
        missed=[]
        for case in saved['terminal_focal_saved']:
            missed.append(dict(case, geometry=by_point[case['seed'],case['full_bank_id']],
                saved_prequery=[r for r in saved['focal_query_states'] if r['seed']==case['seed']]))
        save(output/'summary.json',dict(status='complete', task_count=len(summaries),
            all_task_groups=summaries, first_divergences=first, missed_good_policies=missed,
            direct_component_swaps=saved['direct_component_swaps'],
            source_geometry='source_geometry.json', saved_query_evidence='evidence.json',
            deployed_model_changed=False, limits=plan['limits']))
    attempts=[json.loads(line) for line in (output/'attempts.jsonl').read_text().splitlines()]
    fields=('cpu_seconds','wall_seconds','new_source_calls','new_target_initial_calls',
            'new_target_search_calls','new_verification_calls','new_source_fits',
            'new_truth_function_calls','source_model_restorations','saved_engine_reads',
            'saved_raw_rows_replayed','download_bytes')
    completed=datetime.now(timezone.utc).isoformat()
    save(output/'completion.json',dict(status='complete',
        cost={key:sum(row.get(key,0) for row in attempts) for key in fields},
        attempts=len(attempts), failed_attempts=[r for r in attempts if r['status']!='completed'],
        source_design_parity_max_error=source['summary']['maximum_source_design_error'],
        completed_utc=completed))
    plan.update(status='complete', completed_utc=completed);save(MANIFEST,plan)


if __name__=='__main__':
    main()
