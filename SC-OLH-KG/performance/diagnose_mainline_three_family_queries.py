"""Audit frozen three-family query scores using saved arrays and cached truth only."""
from collections import Counter
from itertools import combinations
import json

import numpy as np
from scipy.stats import norm

from performance.diagnose_mainline_role_choice import MODES, point_record, score_choice
from performance.diagnose_mainline_propagation_attribution import components, RISK, OBJECTIVE
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

MANIFEST = ROOT/'performance/manifests/mainline_three_family_funnel_v1.json'
PARENT = ROOT/'performance/manifests/mainline_three_family_v1.json'
FULL = 'legacy_role_full'
AFTER = 'legacy_role_after_init_point_only'
POINT = 'legacy_role_point_only'
ARMS = (FULL, AFTER, POINT)


def new_good(point, truth, good):
    return None if good is None or point not in truth else point in good


def audit_task(domain, seed, directory, z, failed_arms=()):
    truth = {tuple(row['point']): row['truth'] for row in read(directory/'truth.json')}
    with np.load(directory/'neighbors.npz') as data:
        neighbors = {tuple(p): i for i, p in enumerate(data['points'])}
    initial = {tuple(row['point']) for row in read(directory/'initialization.json')['history']}
    best = min((point for point in initial if truth[point]['true_feasible']),
               key=lambda point: truth[point]['objective'], default=None)
    good = None if best is None else {point for point, t in truth.items() if point not in initial and
            t['true_feasible'] and t['objective'] < truth[best]['objective']-1e-12}
    shared_good = set() if good is None else good & neighbors.keys()
    task = dict(domain=domain, seed=seed, best_initial_neighbor=neighbors.get(best),
        best_initial_truth=truth.get(best), quality_baseline_defined=best is not None,
        unavailable_arms=[a for a in ARMS if a in failed_arms], cached_truth_count=len(truth),
        shared_new_good=None if good is None else [dict(neighbor_index=neighbors[p], truth=truth[p])
                         for p in sorted(shared_good, key=lambda p: neighbors[p])])
    rows, bundles, score_errors = [], {}, []
    for arm in ARMS:
        if arm in failed_arms:
            continue
        case = directory/arm
        with np.load(case/'query_policies.npz') as data:
            points = [tuple(p) for p in data['points']]
        audit = read(case/'query_audit.json')
        history = read(case/'search.json')['history']
        initial_record = None if best is None else dict(id=points.index(best), truth=truth[best])
        states = []
        for stage in audit['stages']:
            score = stage['score']; actions = [points[i] for i in stage['action_ids']]
            choices = {}
            for name, (fk, mk) in MODES.items():
                index, total, mode, count = score_choice(score[fk], score[mk])
                if name == 'actual':
                    error = float(np.max(abs(total-np.asarray(score['total']))))
                    score_errors.append(error)
                    if (error > 1e-12 or index != stage['selected_index'] or
                        stage['action_ids'][index] != stage['selected_id'] or
                        mode != score['sampled_feasibility_mode'] or
                        count != score['sampled_feasible_count'] or
                        actions[index] != tuple(history[stage['paid_prefix']]['point'])):
                        raise ValueError(f'Saved choice mismatch: {domain}/{seed}/{arm}')
                selected = point_record(index, stage, points, neighbors, truth, initial_record, z)
                if best is None:
                    selected.update(feasible_improving=None, true_objective_difference_to_initial=None)
                selected['new_good'] = new_good(actions[index], truth, good)
                choices[name] = dict(mode=mode, feasible_count=count, selected=selected)
            c = components(score, z)
            records = []
            for j, point in enumerate(actions):
                if good is None or point not in good:
                    continue
                record = point_record(j, stage, points, neighbors, truth, initial_record, z)
                record['sampled_objective_rank_all'] = int(1+np.sum(
                    np.asarray(score['sampled_objective']) < record['sampled_objective']))
                record['objective_mean_rank_all'] = int(1+np.sum(
                    np.asarray(score['objective_mean']) < record['objective_mean']))
                if record['actual_selected']:
                    reason = 'selected'
                elif choices['actual']['mode'] == 'minimum_violation':
                    reason = 'minimum_violation_competition'
                elif not record['sampled_feasible']:
                    reason = 'risk_excluded'
                else:
                    reason = 'objective_competition'
                record['actual_selection_reason'] = reason
                # Pointwise truth-error removal diagnoses the known candidate only.
                # Other actions and the saved draw stay unchanged; no new trajectory.
                removals = dict(risk_mean_bias=record['risk_mean_bias'],
                    noise_margin_bias=record['risk_noise_margin_bias'],
                    risk_draw=record['constraint_draw_innovation'])
                record['candidate_only_risk_removal'] = {}
                for label, removed in [*removals.items(),
                    ('mean_and_noise_bias', removals['risk_mean_bias']+removals['noise_margin_bias']),
                    ('all_risk_error', sum(removals.values()))]:
                    margin = np.array(score['sampled_margin']); margin[j] -= removed
                    chosen, _, mode, _ = score_choice(score['sampled_objective'], margin)
                    record['candidate_only_risk_removal'][label] = dict(
                        margin=float(margin[j]), eligible=bool(margin[j] <= 0),
                        candidate_selected=chosen == j, selection_mode=mode)
                records.append(record)
            rows.append(dict(domain=domain, seed=seed, arm=arm,
                quality_baseline_defined=best is not None,
                paid_prefix=stage['paid_prefix'], candidate_count=len(actions),
                cached_truth_action_count=sum(p in truth for p in actions),
                unknown_truth_action_count=sum(p not in truth for p in actions),
                shared_good_present=None if good is None else sum(p in shared_good for p in actions),
                shared_good_absent=None if good is None else [neighbors[p] for p in shared_good if p not in actions],
                good_candidates=None if good is None else records, choices=choices))
            states.append(dict(stage=stage, points=actions, components=c,
                selected=actions[stage['selected_index']]))
        bundles[arm] = dict(states=states, history=history, points=points)
    return task, rows, paired_states(domain, seed, bundles, truth, neighbors, good), score_errors


def paired_states(domain, seed, bundles, truth, neighbors, good):
    modes = [('risk_'+'+'.join(group), group) for size in range(1, 4)
             for group in combinations(RISK, size)]
    modes += [('objective_mean', ('objective_mean',)),
              ('objective_draw', ('objective_draw',)), ('all_objective', OBJECTIVE),
              ('all_components', RISK+OBJECTIVE)]
    rows = []
    for a, b in ((FULL, AFTER), (AFTER, POINT)):
        if a not in bundles or b not in bundles:
            continue
        first = None
        for i, (left, right) in enumerate(zip(bundles[a]['states'], bundles[b]['states'])):
            prefix = left['stage']['paid_prefix']
            changed = left['selected'] != right['selected']
            if changed and first is None:
                first = prefix
            same_history = bundles[a]['history'][:prefix] == bundles[b]['history'][:prefix]
            same_pool = set(left['points']) == set(right['points'])
            def policy(point):
                return dict(neighbor_index=neighbors.get(point), truth=truth.get(point),
                            new_good=new_good(point, truth, good),
                            left_bank_id=bundles[a]['points'].index(point)
                            if point in bundles[a]['points'] else None)
            row = dict(domain=domain, seed=seed, left=a, right=b, paid_prefix=prefix,
                same_paid_history=same_history, same_physical_pool=same_pool,
                query_changed=changed, first_divergence=changed and prefix == first,
                selected={a:policy(left['selected']), b:policy(right['selected'])},
                substitutions=[])
            if same_pool:
                alignment = [right['points'].index(point) for point in left['points']]
                values = {a:left['components'], b:{k:v[alignment]
                    for k,v in right['components'].items()}}
                for receiver, donor in ((a,b), (b,a)):
                    for label, subset in modes:
                        mixed = {k:values[donor if k in subset else receiver][k]
                                 for k in values[a]}
                        index, _, mode, count = score_choice(
                            sum(mixed[k] for k in OBJECTIVE), sum(mixed[k] for k in RISK))
                        chosen = left['points'][index]
                        donor_selected = (left if donor == a else right)['selected']
                        receiver_selected = (left if receiver == a else right)['selected']
                        if label == 'all_components' and chosen != donor_selected:
                            raise ValueError('Aligned full component swap failed to recover donor')
                        row['substitutions'].append(dict(receiver=receiver, donor=donor,
                            substitution=label, selected=policy(chosen), mode=mode,
                            feasible_count=count, donor_choice_recovered=chosen == donor_selected,
                            selection_changed=chosen != receiver_selected))
                row['selected_pair_components'] = []
                for point in dict.fromkeys([left['selected'], right['selected']]):
                    j = left['points'].index(point)
                    row['selected_pair_components'].append(dict(policy=policy(point),
                        arms={arm:{k:float(v[j]) for k,v in cs.items()}
                              for arm,cs in values.items()}))
            else:
                row['substitution_unavailable_reason'] = 'Different physical candidate pools after adaptation'
            rows.append(row)
    return rows


def summarize(plan):
    tasks, states, pairs, errors = [], [], [], []
    for family in plan['families']:
        search = read(ROOT/family['search_manifest'])
        z = float(norm.ppf(1-search['task']['alpha']))
        for seed in search['target_seeds']:
            result = audit_task(family['domain'], seed,
                ROOT/search['output_dir']/f'seed_{seed}', z,
                failed_arms=[r['arm'] for r in search.get('execution_failures', []) if r['seed']==seed])
            task, rows, paired, err = result
            tasks.append(task); states.extend(rows); pairs.extend(paired); errors.extend(err)
    summary = []
    for domain in [f['domain'] for f in plan['families']]:
        for arm in ARMS:
            selected = [r for r in states if r['domain'] == domain and r['arm'] == arm]
            exposures = [p for r in selected for p in (r['good_candidates'] or [])]
            summary.append(dict(domain=domain, arm=arm, recorded_states=len(selected),
                quality_baseline_undefined_states=sum(not r['quality_baseline_defined'] for r in selected),
                good_candidate_state_exposures=len(exposures),
                good_candidates_selected=sum(p['actual_selected'] for p in exposures),
                reasons=dict(Counter(p['actual_selection_reason'] for p in exposures)),
                candidate_only_removal={mode:dict(
                    eligible=sum(p['candidate_only_risk_removal'][mode]['eligible'] for p in exposures),
                    selected=sum(p['candidate_only_risk_removal'][mode]['candidate_selected'] for p in exposures))
                    for mode in ('risk_mean_bias','noise_margin_bias','risk_draw',
                                 'mean_and_noise_bias','all_risk_error')},
                saved_score_modes={mode:dict(
                    changed=sum(r['choices'][mode]['selected']['policy_id'] !=
                                r['choices']['actual']['selected']['policy_id'] for r in selected),
                    selected_new_good=sum(r['choices'][mode]['selected']['new_good'] is True for r in selected),
                    unknown_new_good=sum(r['choices'][mode]['selected']['new_good'] is None for r in selected),
                    unknown_truth=sum(r['choices'][mode]['selected']['truth'] is None for r in selected))
                    for mode in MODES}))
    return dict(status='complete', tasks=tasks, states=states, paired_states=pairs,
        summary=summary, reconstruction=dict(recorded_choices=len(errors), max_score_error=max(errors)),
        limits=('All registered tasks and available role arms; registered failed arms remain unavailable. '
          'A missing feasible initial baseline leaves all quality-gain labels undefined. Good means outside shared initial10 and better '
          'than its best cached true-feasible member. Shared64 opportunity truth is complete; '
          'unqueried dynamic policies have partial cached truth. Candidate-state exposures repeat '
          'policies and are not independent task samples. Saved-score substitutions are fixed-state '
          'diagnoses, not optimizer outcomes; truth-error removal is oracle and never a model fix. '
          'FULL/AFTER and AFTER/SOURCE0 component swaps isolate same-history comparisons only before '
          'their first query divergence; later differing histories are descriptive. Risk draw denotes '
          'the saved realized TS innovation, not all epistemic uncertainty.'))


def main():
    output = ROOT/read(MANIFEST)['output_dir']
    with attempt(output, 'saved-query-funnel-audit') as cost:
        cost.update(new_target_initial_calls=0, new_source_fits=0,
            new_truth_function_calls=0, posterior_restorations=0, download_bytes=0)
        result = summarize(read(PARENT))
        save(output/'query_diagnosis.json', result)
    attempts = [json.loads(line) for line in (output/'attempts.jsonl').read_text().splitlines()
                if json.loads(line)['stage'] == 'saved-query-funnel-audit']
    fields = ('cpu_seconds','wall_seconds','new_source_calls','new_target_initial_calls',
              'new_target_search_calls','new_verification_calls','new_source_fits',
              'new_truth_function_calls','posterior_restorations','download_bytes')
    save(output/'query_completion.json', dict(status='complete',
        reconstruction=result['reconstruction'],
        cost={k:sum(row.get(k,0) for row in attempts) for k in fields},
        failed_attempts=[r for r in attempts if r['status'] != 'completed']))
    print(json.dumps(result['reconstruction']))


if __name__ == '__main__':
    main()
