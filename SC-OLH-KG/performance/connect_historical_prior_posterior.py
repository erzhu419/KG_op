"""Connect restored source maps to existing coefficient and raw-field filters."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acquisition.decision_backends import evaluate_or_replicate_action_set
from core.budgeted_guarded_kg import BudgetedTerminalActions
from core.gpr import posterior_mixture_weights
from core.raw_likelihood_lookahead import likelihood_query_lookahead
from core.terminal_verification import select_initial_empirical_objective_incumbent
from performance.adapt_historical_source_structure import restore_selector
from performance.benchmark_frozen_proposal_only import _select_shortlist
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.replay_guarded_deployment import shortlist_indices
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture
from representation.meta_prior import LearnedMetaPrior
from representation.source_function_prior import SourceFunctionPrior
from representation.transferable_spectral import SourceDomainBatch

PLAN = 'performance/manifests/historical_prior_posterior_connection_v1.json'
HEADS = ('f', 'g', 'eta_g', 'eta_f')


def source_prior(basis, coordinates, tasks, head, meta):
    """Reuse the finite coefficient law in source-standardized response units.

    Its existing 0.25/0.02 floors are in normalized units. They must scale
    back with the covariance, rather than becoming physical variance floors.
    This historical reference uses Gaussian source labels; it is not the
    current exact raw-source covariance learning law.
    """
    y = np.concatenate([np.asarray(t['response'])[:, head] for t in tasks])
    noise = [np.asarray(t['noise'])[:, head] for t in tasks]
    center, scale = float(y.mean()), float(y.std())
    batches = [SourceDomainBatch(t['name'], psi,
        (np.asarray(t['response'])[:, head] - center) / scale)
        for t, psi in zip(tasks, coordinates)]
    prior = SourceFunctionPrior.from_basis(batches, [v / scale**2 for v in noise], basis, 'learned')
    prior.mean *= scale; prior.mean[0] += center
    prior.covariance *= scale**2; prior.discrepancy_variance *= scale**2
    evidence = [{'domain': t['name'], **meta.spectral_coefficient_statistics(
        basis.transform(psi), np.asarray(t['response'])[:, head], np.ones(len(psi)))}
        for t, psi in zip(tasks, coordinates)]
    hyper = meta._aggregate_spectral_coefficient_prior(evidence)
    return prior, hyper, {'center': center, 'scale': scale}


def field_moments(prior, coordinates):
    design = np.column_stack([np.ones(len(coordinates)), prior.basis.transform(coordinates)])
    return design @ prior.mean, design @ prior.covariance @ design.T + prior.discrepancy_variance * np.eye(len(design))


def paired_field(mean_head, risk_head):
    n = len(mean_head[0]); mean = np.column_stack([mean_head[0], risk_head[0]])
    covariance = np.zeros((n, 2, n, 2))
    covariance[:, 0, :, 0] = mean_head[1]; covariance[:, 1, :, 1] = risk_head[1]
    return JointFunctionRiskPosterior.from_moments(mean, covariance)


def snapshot(model):
    return {'names': model.names, 'weights': model.weights.tolist(), 'paid_episodes': model.paid_episodes,
        'history': list(model.history), 'components': [{'f_mean': f.predict(full_cov=True)[0].tolist(),
        'f_covariance': f.predict(full_cov=True)[1].tolist(), 'g_mean': g.predict(full_cov=True)[0].tolist(),
        'g_covariance': g.predict(full_cov=True)[1].tolist()} for f, g in model.components]}


def run():
    started = time.process_time(); spec = read(PLAN); selection = read(spec['selection_plan'])
    maps = read(spec['maps'])['rows']; fields = read(selection['raw_source_fields'])['rows']
    laws = read(selection['risk_laws'])['rows']; raw = read(spec['source_data']); public = read(spec['public_inputs'])
    meta = LearnedMetaPrior(); priors, models, checks = [], [], []
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    for layout, path in selection['source_inputs_by_layout'].items():
        inputs = read(path); tasks = [t for t in inputs['source_tasks'] if t['name'] in selection['source_tasks']]
        field = next(r for r in fields if r['layout'] == layout)
        law = next(r for r in laws if (r['layout'], r['arm']) == (layout, field['active_arm']))
        mean_coordinates = [np.asarray(t['psi']) for t in tasks]
        reference = np.asarray(law['public_reference_psi'])
        risk_coordinates = np.split(reference[[s['policy'] for s in field['sites']]], len(tasks))
        query = next(t for t in inputs['source_tasks'] if t['name'] == spec['development_task'])
        indices = inputs['execution']['source_site_indices_by_task'][spec['development_task_index']][:spec['candidate_count']]
        risk_inputs = read(spec['query_risk_inputs_by_layout'][layout])
        np.testing.assert_allclose(reference, risk_inputs['public_reference_psi'], atol=1e-12, rtol=0.)
        assert risk_inputs['training_task_ids'] == [2, 3, 4, 5]
        query_mean = np.asarray(query['psi'])[:len(indices)]
        query_risk = np.asarray(risk_inputs['heldout_public_psi'])[indices]
        old_query_risk = np.asarray(next(t for t in inputs['risk_source_tasks'] if t['name'] == query['name'])['psi'])
        old_map_difference = float(np.max(abs(old_query_risk - query_risk)))
        candidates = [tuple(raw['bank'][i]) for i in indices]
        observations = np.asarray([raw['tasks'][spec['development_task_index']]['Y_replicates'][i][0]
            for i in indices[:spec['paid_prefix']]])[:, :2]
        observations[:, 1] -= raw['tasks'][spec['development_task_index']]['tau']
        names, components = [], []
        for kind in ('spectral', 'additive'):
            for support in ('head_specific', 'joint'):
                moments = []
                for head, name in enumerate(HEADS):
                    coordinates = ([np.column_stack([m, v]) for m, v in zip(mean_coordinates, risk_coordinates)]
                        if support == 'joint' else mean_coordinates if head < 2 else risk_coordinates)
                    target = (np.column_stack([query_mean, query_risk]) if support == 'joint'
                        else query_mean if head < 2 else query_risk)
                    record = next(r for r in maps if (r['layout'], r['selector'], r['head']) ==
                        (layout, kind, 'joint_f_g_eta_g_eta_f' if support == 'joint' else name))
                    basis = restore_selector(record, selection)
                    prior, hyper, units = source_prior(basis, coordinates, tasks, head, meta)
                    model = prior.target_model(target)
                    np.testing.assert_allclose(model.posterior_mean_many([(i,) for i in range(len(target))]),
                        field_moments(prior, target)[0], atol=1e-12, rtol=0.)
                    sparse = None
                    if head < 2:
                        source_noise = float(np.mean(np.concatenate([np.asarray(t['noise'])[:, head] for t in tasks]))) * 3
                        model.enable_adaptive_sparsity({'source_pip': hyper['prior_pip'], 'source_slab_scale': hyper['slab_scale']},
                            [(i,) for i in range(len(observations))], observations[:, head], [source_noise]*len(observations),
                            deviation_variance=prior.discrepancy_variance)
                        sparse = model.adaptive_sparsity_diagnostics()
                    moments.append(field_moments(prior, target))
                    priors.append({'layout': layout, 'selector': kind, 'support': support, 'head': name,
                        'source_tasks': selection['source_tasks'], 'source_record_count': 48,
                        'feature_count': basis.feature_dim, 'response_units': units,
                        'mean': prior.mean.tolist(), 'covariance': prior.covariance.tolist(),
                        'discrepancy_variance': prior.discrepancy_variance,
                        'historical_coefficient_hyperprior': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in hyper.items()},
                        'gaussian_sparse_reference': sparse})
                names.append(kind + '_' + support)
                components.append((paired_field(moments[0], moments[3]), paired_field(moments[1], moments[2])))
        mixture = JointResponseMixture(names, components)
        prior_state = snapshot(mixture)
        for i, value in enumerate(observations):
            before = mixture.weights.copy(); result = mixture.update_raw(i, value)
            _, expected = posterior_mixture_weights(before, result['joint_log_scores'])
            np.testing.assert_allclose(mixture.weights, expected, atol=1e-12, rtol=0.)
        problem = problem_for(public['plan'], public['tasks'][spec['development_task_index']])
        initial, _ = _select_shortlist(problem, candidates[:len(observations)], observations)
        positions = [candidates.index(tuple(r['point'])) for r in initial]
        objective_incumbent = select_initial_empirical_objective_incumbent(candidates[:len(observations)],
            observations, n0=len(observations))
        positions[0] = candidates.index(tuple(objective_incumbent['point']))
        table, incumbent = shortlist_indices(positions, list(range(len(observations))), observations,
            len(candidates), range(len(candidates)))
        protocol = dict(spec['terminal_protocol'], objective_reference=float(np.dot(problem.weights, problem.ref_point)),
            chance_alpha=problem.alpha)
        terminal = BudgetedTerminalActions(table, incumbent, protocol)
        values, costs, actions = mixture.terminal_values(terminal, spec['total_target_call_cap']-len(observations),
            spec['terminal_draw_power'], spec['seed'])
        priority = np.full(len(candidates), np.inf)
        for value, action in zip(values, actions):
            priority[action['primary']] = min(priority[action['primary']], -float(value))
        observed = [(candidates[i], v) for i, v in enumerate(observations)]
        action_sets = {str(count): evaluate_or_replicate_action_set(candidates, observed, problem,
            iteration=len(observations), seed=spec['seed'], new_action_count=count,
            new_action_policy='canonical_plus_posterior', new_action_priority=priority)
            for count in (1, 4)}
        one, four = [action_sets[str(k)] for k in (1, 4)]
        assert set(one['active_indices']) <= set(four['active_indices'])
        assert np.array_equal(one['replicate_mask'], four['replicate_mask'])
        assert len(four['new_action_indices']) == 4
        record = {'layout': layout, 'development_task': query['name'], 'candidate_bank_indices': indices,
            'paid_observations': observations.tolist(), 'paid_variance_labels': 0,
            'old_query_risk_coordinate_max_difference': old_map_difference, 'current_query_risk_map_used': True,
            'prior': prior_state, 'posterior': snapshot(mixture), 'terminal_protocol': protocol,
            'terminal_values': values.tolist(), 'terminal_costs': costs.tolist(), 'terminal_actions': actions,
            'terminal_priorities': priority.tolist(), 'action_sets': {k: {key: v.tolist() if isinstance(v, np.ndarray) else v
                for key, v in a.items()} for k, a in action_sets.items()},
            'balanced4_contains_one_new_set': True, 'eligible_repeats_preserved': True}
        models.append(record)
        checks.append({'layout': layout, 'raw_updates': len(observations), 'shared_identity_weight_update_passed': True,
            'maximum_raw_integration_relative_error': max(field.max_integration_relative_error for pair in mixture.components for field in pair),
            'balanced4_new_count': len(four['new_action_indices']), 'repeat_count': int(four['replicate_mask'].sum())})
        save(out / 'source_priors.json', {'execution': spec, 'rows': priors})
        save(out / 'states.json', {'execution': spec, 'rows': models})
        print('connected', layout, 'prior heads', len(priors), 'weights', mixture.weights, 'actions', four['active_indices'], flush=True)
        if layout == spec['lookahead_layout']:
            lookahead = []
            for index in four['active_indices']:
                if time.process_time()-started >= spec['maximum_single_thread_cpu_seconds']:
                    break
                result = likelihood_query_lookahead(mixture, terminal, spec['total_target_call_cap']-len(observations), int(index),
                    draw_power=spec['lookahead_draw_power'], fantasy_power=spec['fantasy_power'], seed=spec['seed'])
                lookahead.append(result)
                save(out / 'lookahead.json', {'execution': spec, 'layout': layout, 'rows': lookahead,
                    'all_balanced4_actions_evaluated': len(lookahead) == len(four['active_indices'])})
                print('raw lookahead', int(index), 'net', result['net_reward_of_paid_episode'],
                    'ESS', result['effective_sample_size_min'], flush=True)
    save(out / 'results.json', {'execution': spec, 'checks': checks, 'source_prior_heads': len(priors),
        'complete_hypotheses': sum(len(r['posterior']['names']) for r in models),
        'cpu_seconds': time.process_time()-started, 'cost': {'new_selector_fits': 0, 'coefficient_prior_fits': len(priors),
            'new_source_chains': 0, 'new_simulator_calls': 0, 'new_verifier_calls': 0, 'independent_targets_opened': 0},
        'no_model_adopted': True, 'independent_confirmation': False,
        'limitations': 'Historical Gaussian-label finite coefficient reference,source-standardized floors and old inclusion evidence;task coefficient second moments are not fit-noise-deconvolved. Raw paid filters use the exact single-episode density with subsequent Gaussian moment projection. The sparse Gaussian reference is separate from the raw filter. All priors use restored source maps;task0 is opened development. Eight cached policies and low-order numerical rules test interfaces,not full-bank decision advantage or current96-coefficient covariance learning.'})


if __name__ == '__main__':
    run()
