"""Trace terminal omissions from frozen Stage41 JSON/NPZ evidence only."""
from collections import Counter

import numpy as np

from performance.diagnose_mainline_role_terminal import policy_key
from performance.run_mainline_legacy_backend_bridge import ROOT, read

FULL = 'legacy_role_full'
LOCAL = 'legacy_role_after_init_point_only'
ARMS = (FULL, LOCAL)
SUFFIX = '_native_terminal'


def raw_summary(history, point, truth):
    rows = [row['raw'] for row in history if policy_key(row['point']) == point]
    values = np.asarray(rows, dtype=float)
    count = len(values)
    means = np.mean(values, axis=0)
    return dict(count=count, raw=rows, objective_mean=float(means[0]),
        constraint_mean_centered=float(means[1]),
        sample_standard_error=(np.std(values, axis=0, ddof=1) / np.sqrt(count)).tolist()
            if count > 1 else None,
        standard_error_scope='Descriptive within-policy estimate; unavailable for one observation.',
        objective_mean_error=float(means[0] - truth['objective']),
        constraint_mean_error=float(means[1] - truth['constraint_mean_centered']),
        constraint_draw_standardized_errors=(
            (values[:, 1] - truth['constraint_mean_centered']) / truth['constraint_sigma']).tolist(),
        standardized_error_scope='Post-hoc cached true sigma; not an estimated search noise parameter.')


def role_eligibility(point, terminal):
    """The saved feasible-first ranks can identify gates without restoring a model."""
    count = terminal['objective_challenger']['eligible_count']
    ranked = {row['posterior_rank']: policy_key(row['point'])
              for row in terminal['frozen_shortlist'] if row.get('posterior_rank') is not None}
    rank = next((rank for rank, key in ranked.items() if key == point), None)
    if rank is not None:
        eligible, basis = rank <= count, 'saved_feasible_first_rank_and_eligible_count'
    elif count == 0:
        eligible, basis = False, 'saved_zero_eligible_count'
    elif set(range(1, count + 1)).issubset(ranked):
        eligible, basis = False, 'all_eligible_ranks_belong_to_other_saved_policies'
    else:
        eligible, basis = None, 'omitted_candidate_final_components_not_saved'
    return dict(eligible=eligible, basis=basis, eligible_count=count,
        known_ranks=sorted(ranked), posterior_rank=rank,
        probability_violation=None, final_objective_mean=None)


def snapshot_evidence(directory, point, initial, truth):
    audit = read(directory / 'query_audit.json')
    bank = [policy_key(row) for row in np.load(directory / 'query_policies.npz')['points']]
    identifiers = {point: bank.index(point), initial: bank.index(initial)}
    records = []
    for stage in audit['stages']:
        if any(index not in stage['scored_ids'] for index in identifiers.values()):
            continue
        i, j = [stage['scored_ids'].index(identifiers[key]) for key in (point, initial)]
        score = stage['score']
        fields = ('objective_mean', 'constraint_mean', 'constraint_epistemic',
                  'constraint_aleatoric', 'stochastic_margin_mean', 'probability_violation')
        row = {key: score[key][i] for key in fields}
        row.update(paid_prefix=stage['paid_prefix'],
            objective_difference_to_initial=score['objective_mean'][i] - score['objective_mean'][j],
            constraint_mean_error=score['constraint_mean'][i] - truth['constraint_mean_centered'],
            chance_margin_error=score['stochastic_margin_mean'][i] - truth['chance_margin'],
            epistemic_standard_deviation=float(np.sqrt(score['constraint_epistemic'][i])),
            predicted_noise_sigma=float(np.sqrt(score['constraint_aleatoric'][i])))
        row['noise_quantile_error'] = (row['chance_margin_error'] - row['constraint_mean_error'])
        records.append(row)
    return dict(scope='Saved pre-query prefixes 10/11/12, not the final 13-observation terminal.',
                records=records)


def analyze(plan):
    source, terminals = ROOT / plan['source_output_dir'], ROOT / plan['terminal_output_dir']
    cases, focused = [], []
    for seed in plan['target_seeds']:
        task = source / f'seed_{seed}'
        truth = {policy_key(row['point']): row['truth'] for row in read(task / 'truth.json')}
        neighbors = {policy_key(row): i for i, row in enumerate(np.load(task / 'neighbors.npz')['points'])}
        searches = {arm: read(task / arm / 'search.json') for arm in ARMS}
        initial = policy_key(searches[FULL]['history'][0]['point'])
        initial_keys = [policy_key(row['point']) for row in searches[FULL]['history'][:10]]
        query_keys = {arm: list(dict.fromkeys(policy_key(row['point'])
                     for row in search['history'][10:])) for arm, search in searches.items()}
        good = {arm: {key for key in keys if truth[key]['true_feasible'] and
                     truth[key]['objective'] < truth[initial]['objective'] - 1e-12}
                for arm, keys in query_keys.items()}

        def label(key):
            if key in neighbors:
                return f'neighbor:{neighbors[key]}'
            if key in initial_keys:
                return f'initial:{initial_keys.index(key)}'
            return next(f'{arm}:history:{i}' for arm, search in searches.items()
                        for i, row in enumerate(search['history']) if policy_key(row['point']) == key)

        task_cases = {}
        for arm, search in searches.items():
            native = read(task / arm / 'native_terminal.json')
            native_records = {policy_key(row['point']): row for row in native['observed']}
            for backend, frozen, cell in (('role', search, arm), ('native', native, arm + SUFFIX)):
                verified = read(terminals / f'seed_{seed}' / cell / 'original/verification.json')
                terminal, verification = frozen['terminal'], verified['verification']
                deployed = policy_key(verified['deployed'])
                attempts = {row['candidate_index'] + 1: row for row in verification['attempts']}
                shortlist = []
                for entry in terminal['frozen_shortlist']:
                    position, key = entry['shortlist_position'], policy_key(entry['point'])
                    certificate = attempts.get(position)
                    shortlist.append(dict(policy=label(key), position=position,
                        role=entry['shortlist_role'], posterior_rank=entry.get('posterior_rank'),
                        candidate_universe=entry.get('candidate_universe'),
                        selection_contract=entry.get('selection_contract'),
                        recorded_probability_violation=entry.get('selected_posterior_violation_probability',
                            entry.get('selected_probability_violation')),
                        improving=key in good[arm], true_feasible=truth[key]['true_feasible'],
                        safety_attempted=certificate is not None,
                        safety_passed=certificate['certified'] if certificate else None,
                        upper_margin=certificate['upper_margin'] if certificate else None,
                        deployed=key == deployed and verification['certified']))
                by_key = {policy_key(entry['point']): row
                          for entry, row in zip(terminal['frozen_shortlist'], shortlist)}
                policies = []
                for key in query_keys[arm]:
                    entry = by_key.get(key)
                    if entry is None:
                        disposition = 'not_on_frozen_shortlist'
                    elif not entry['safety_attempted']:
                        disposition = 'not_verified_after_earlier_terminal_decision'
                    elif not entry['safety_passed']:
                        disposition = 'safety_certificate_failed'
                    elif entry['deployed']:
                        disposition = 'deployed'
                    else:
                        disposition = 'objective_guard_retained_incumbent'
                    policies.append(dict(policy=label(key), feasible_improving=key in good[arm],
                        disposition=disposition, shortlist_position=entry['position'] if entry else None,
                        quality_gain=truth[key]['quality'] - truth[initial]['quality']))
                improving = [row for row in policies if row['feasible_improving']]
                result = dict(seed=seed, history_arm=arm, terminal_backend=backend,
                    queries=[label(policy_key(row['point'])) for row in search['history'][10:]],
                    primary=label(policy_key(frozen['primary'])), initial=label(initial),
                    deployed=label(deployed), certified=verification['certified'],
                    objective_guard_status=verification['objective_comparison']['status'],
                    objective_guard_calls=verification['objective_comparison']['simulation_calls'],
                    actual_verification_calls=verified['new_verification_calls'],
                    objective_challenger=terminal['objective_challenger'], shortlist=shortlist,
                    queried_policies=policies, improving_funnel=dict(
                        discovered=len(improving), shortlisted=sum(row['shortlist_position'] is not None for row in improving),
                        safety_attempted=sum(row['improving'] and row['safety_attempted'] for row in shortlist),
                        safety_passed=sum(row['improving'] and row['safety_passed'] is True for row in shortlist),
                        deployed=sum(row['disposition'] == 'deployed' for row in improving)))
                cases.append(result)
                task_cases[arm, backend] = result
            if arm == FULL:
                for key in sorted(good[FULL] - set(query_keys[LOCAL]), key=label):
                    observed = native_records[key]
                    objective_difference = observed['objective'] - native_records[initial]['objective']
                    detail = dict(seed=seed, policy=label(key), truth=truth[key], initial_truth=truth[initial],
                        query_history_indices=[i for i, row in enumerate(search['history']) if policy_key(row['point']) == key],
                        raw=raw_summary(search['history'], key, truth[key]),
                        initial_raw=raw_summary(search['history'], initial, truth[initial]),
                        role_final=role_eligibility(key, search['terminal']),
                        native_final={k: v for k, v in observed.items() if k != 'point'},
                        native_objective_difference_to_initial=objective_difference,
                        native_initial={k: v for k, v in native_records[initial].items() if k != 'point'},
                        native_primary_failure=('posterior_risk_gate' if not observed['eligible'] else
                            'posterior_objective_order' if objective_difference > 0 else 'other_ranked_candidate'),
                        role_prequery=snapshot_evidence(task / arm, key, initial, truth[key]),
                        terminal_exposure={backend: dict(
                            disposition=next(row['disposition'] for row in task_cases[arm, backend]['queried_policies'] if row['policy'] == label(key)),
                            shortlist=task_cases[arm, backend]['shortlist'],
                            objective_guard_status=task_cases[arm, backend]['objective_guard_status'])
                            for backend in ('role', 'native')})
                    detail['raw_objective_difference_to_initial'] = detail['raw']['objective_mean'] - detail['initial_raw']['objective_mean']
                    focused.append(detail)

    aggregates = []
    for arm in ARMS:
        for backend in ('role', 'native'):
            group = [row for row in cases if row['history_arm'] == arm and row['terminal_backend'] == backend]
            dispositions = Counter(row['disposition'] for case in group for row in case['queried_policies'] if row['feasible_improving'])
            aggregates.append(dict(history_arm=arm, terminal_backend=backend, tasks=len(group),
                improving_funnel={key: sum(row['improving_funnel'][key] for row in group)
                                  for key in group[0]['improving_funnel']},
                improving_policy_dispositions=dict(dispositions)))
    return dict(status='complete', task_count=len(plan['target_seeds']), cases=cases, arms=aggregates,
        full_only_good_policies=focused, full_only_good_policy_count=len(focused),
        policy_identity='Exact physical vectors; zero-based neighbor/initial indices; one-based shortlist positions.',
        scope='Saved JSON/NPZ and cached truth only; zero posterior restores, model changes, simulations or new truth calls.',
        limitations=['Final role probabilities/means for omitted policies were not saved. Eligibility is inferred only when saved eligible count and feasible-first ranks determine it.',
                     'Prefix12 predictions precede the final observation and are not final terminal predictions.',
                     'Unexposed good policies have no terminal safety/objective certificate; cached truth does not supply a counterfactual certificate.'])
