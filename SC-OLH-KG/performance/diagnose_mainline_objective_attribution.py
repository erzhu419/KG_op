"""Attribute frozen objective differences to node updates and mixture weights."""
import argparse
import json
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp, ndtr, softmax

from performance.diagnose_mainline_initialization import stream_head
from performance.mainline_role_prior import role_inputs, role_moment_state, _history
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, load_pickle, read, save

MANIFEST = ROOT/'performance/manifests/mainline_objective_attribution_v1.json'


def collect_nodes(record, D, E, history, selected):
    """Read projected nodes while the original filter performs its normal update."""
    shape = (len(history)+1, len(record['mean']))
    result = dict(log_mass=np.empty(shape), mean=np.empty((*shape, len(selected))),
        delta_variance=np.empty((*shape, len(selected)-1)),
        observation_mean=np.zeros(shape),
        delta_observation_covariance=np.zeros((*shape, len(selected)-1)))

    def observe(prefix, start, mass, field, LF, unobserved, local):
        count = len(field.means)
        index = (prefix, slice(start, start+count))
        projection = LF[selected]
        contrast = projection[1:]-projection[0]
        result['log_mass'][index] = mass+field.log_weights
        result['mean'][index] = field.means[:, 0::2]@projection.T
        result['delta_variance'][index] = np.einsum('ai,kij,aj->ka', contrast,
            field.covariances[:, 0::2, 0::2], contrast) + record['law']['white']*(
                unobserved[selected[1:]].astype(int)+int(unobserved[selected[0]]))
        if prefix < len(history):
            position = local[history[prefix][0]]
            result['observation_mean'][index] = field.means[:, 2*position]
            result['delta_observation_covariance'][index] = (
                field.covariances[:, 0::2, 2*position]@contrast.T)

    trajectory, events = stream_head(record, D, E, history, 0, 64, observer=observe)
    return result, trajectory, events


def attribute(nodes, head_mass, prior):
    logs = nodes['log_mass']
    q = np.exp(logs-logsumexp(logs, axis=2, keepdims=True))
    w = softmax(prior[:, None]+head_mass.sum(axis=2), axis=0)
    mu = nodes['mean']
    identity_mean = np.einsum('itk,itka->ita', q, mu)
    means = np.einsum('it,ita->ta', w, identity_mean)
    deltas = mu[..., 1:]-mu[..., :1]
    delta_mean = means[:, 1:]-means[:, :1]
    second = np.einsum('it,itk,itka->ta', w, q,
                        nodes['delta_variance']+deltas**2)
    variance = second-delta_mean**2
    probability = np.einsum('it,itk,itka->ta', w, q,
        ndtr(-deltas/np.sqrt(np.maximum(nodes['delta_variance'], 1e-300))))
    f_only_w = softmax(prior[:, None]+head_mass[:, :, 0], axis=0)
    f_only_means = np.einsum('it,ita->ta', f_only_w, identity_mean)
    steps = []
    for t in range(mu.shape[1]-1):
        old_w, old_q = w[:, t], q[:, t]
        new_q, old_mu, new_mu = q[:, t+1], mu[:, t], mu[:, t+1]
        means_new = identity_mean[:, t+1]
        terms = dict(
            within_node=np.einsum('i,ik,ika->a', old_w, old_q, new_mu-old_mu),
            node_reweight=np.einsum('i,ik,ika->a', old_w, new_q-old_q, new_mu))
        increments = head_mass[:, t+1]-head_mass[:, t]
        old_log = prior+head_mass[:, t].sum(axis=1)
        after_f, after_g = softmax(old_log+increments[:, 0]), softmax(old_log+increments[:, 1])
        terms['identity_f'] = .5*(after_f-old_w+w[:, t+1]-after_g)@means_new
        terms['identity_g'] = .5*(after_g-old_w+w[:, t+1]-after_f)@means_new
        error = float(np.max(abs(sum(terms.values())-(means[t+1]-means[t]))))
        if error > 1e-10:
            raise ValueError('Objective attribution does not telescope')
        steps.append(dict(before=t, after=t+1, point_contributions=terms,
            difference_contributions={k:v[1:]-v[0] for k,v in terms.items()},
            reconstruction_error=error))
    obs_mean = np.einsum('it,itk,itk->t', w, q, nodes['observation_mean'])
    cross = np.einsum('it,itk,itka->ta', w, q,
        nodes['delta_observation_covariance']+deltas*nodes['observation_mean'][..., None])
    cross -= delta_mean*obs_mean[:, None]
    return dict(mean=means, difference_mean=delta_mean, difference_variance=variance,
        probability_improving=probability, identity_weights=w.T,
        f_only_identity_difference=f_only_means[:, 1:]-f_only_means[:, :1],
        observation_predictive_mean=obs_mean[:-1],
        difference_observation_covariance=cross[:-1], steps=steps)


def run_case(plan, seed):
    parent = ROOT/plan['input_dir']/f'seed_{seed}'
    output = ROOT/plan['output_dir']/f'seed_{seed}'
    parent_plan = read(ROOT/plan['parent_manifest'])
    case = parent/'legacy_role_full'
    with attempt(output, 'objective-node-attribution', 'FactorShockStatePolicyRZDT1') as cost:
        cost.update(new_target_initial_calls=0, new_source_fits=0, new_truth_function_calls=0,
                    download_bytes=0, source_model_restorations=1, saved_raw_rows_replayed=12)
        engine = load_pickle(case/'search_engine.pkl')
        bundle = load_pickle(ROOT/parent_plan['frozen_assets']['source_models_file'])
        rows = read(case/'search.json')['history'][:12]
        neighbors = np.load(parent/'neighbors.npz')['points']
        focus = plan['focus_neighbors'][str(seed)]
        points = [tuple(rows[0]['point'])]+[tuple(neighbors[i]) for i in focus]
        bank = list(dict.fromkeys([tuple(row['point']) for row in rows]+points))
        selected = [bank.index(point) for point in points]
        history = _history(engine, bank, rows)
        all_nodes, heads, events = [], [], []
        with ExitStack() as stack:
            for name in ('simulate', 'true_objective', 'true_constraint_mean', 'true_sigma'):
                stack.enter_context(patch.object(engine.problem, name, side_effect=AssertionError(name)))
            replacement, designs = role_inputs(bundle, engine, bank)
            for identity, pair, design in zip(bundle['identities'], replacement['components'], designs):
                nodes, f, f_events = collect_nodes(pair[0], *design[0], history, selected)
                g, g_events = stream_head(pair[1], *design[1], history, 1, 64)
                all_nodes.append(nodes); heads.append((f, g))
                events.extend(dict(event, identity=identity) for event in f_events+g_events)
                print(seed, identity, 'complete', flush=True)
            stacked = {key:np.stack([n[key] for n in all_nodes]) for key in all_nodes[0]}
            head_mass = np.array([[[f[0], g[0]] for f,g in zip(*pair)] for pair in heads])
            prior = np.asarray(bundle['log_weights'])
            summary = attribute(stacked, head_mass, prior)
            audit = read(case/'query_audit.json')['stages']
            policies = np.load(case/'query_policies.npz')['points']
            parity = []
            for stage in audit:
                t = stage['paid_prefix']
                position = {tuple(policies[i]):j for j,i in enumerate(stage['action_ids'])}
                a = [position[p] for p in points]
                weights = summary['identity_weights'][t]
                m = np.array([np.column_stack([f[t][1],g[t][1]])[selected] for f,g in heads])
                v = np.array([np.column_stack([f[t][2],g[t][2]])[selected] for f,g in heads])
                noise = np.array([np.column_stack([f[t][3],g[t][3]])[selected] for f,g in heads])
                mean = np.einsum('i,ipj->pj', weights, m)
                np.testing.assert_allclose(summary['mean'][t], mean[:,0], rtol=1e-8,
                                           atol=1e-10, err_msg='node mean projection')
                var = np.einsum('i,ipj->pj', weights, v+(m-mean)**2)
                eta = np.einsum('i,ipj->pj', weights, noise)
                predicted = dict(objective_mean=mean[:,0], constraint_mean=mean[:,2],
                    objective_epistemic=var[:,0], constraint_epistemic=var[:,2],
                    constraint_aleatoric=eta[:,1])
                errors = {}
                for key, value in predicted.items():
                    expected = np.asarray(stage['score'][key])[a]
                    np.testing.assert_allclose(value, expected, rtol=1e-8, atol=1e-10, err_msg=key)
                    errors[key] = float(np.max(abs(value-expected)))
                parity.append(dict(paid_prefix=t, errors=errors))
            covariance_error = None
            if seed == int(next(iter(plan['focus_neighbors']))):
                reference = role_moment_state(bundle, engine, bank, rows[:10])
                cost['source_model_restorations'] += 1
                cost['saved_raw_rows_replayed'] += 10
                events.extend(reference['roundoff_events'])
                cov = reference['cov'][:,0,:,0]
                expected = [cov[i,i]+cov[selected[0],selected[0]]-2*cov[i,selected[0]] for i in selected[1:]]
                np.testing.assert_allclose(summary['difference_variance'][10], expected, rtol=1e-8, atol=1e-10)
                covariance_error = float(np.max(abs(summary['difference_variance'][10]-expected)))
        np.savez_compressed(output/'node_projections.npz', **stacked, head_mass=head_mass,
                            source_log_weights=prior)
        save(output/'projection.json', dict(seed=seed, focus_neighbors=focus,
            identity_names=bundle['identities'], summary=summary, parity=parity,
            difference_covariance_parity=covariance_error, roundoff_events=events))
        # Outcomes are opened only after the computed projections have been saved.
        truth = {tuple(r['point']):r['truth'] for r in read(parent/'truth.json')}
        focus_truth = [truth[p] for p in points]
        periods = []
        for begin,end in ((0,10),(10,11),(11,12),(10,12)):
            terms = {key:sum(s['difference_contributions'][key] for s in summary['steps'][begin:end])
                     for key in summary['steps'][0]['difference_contributions']}
            periods.append(dict(begin=begin,end=end,difference_contributions=terms))
        save(output/'diagnosis.json', dict(seed=seed, focus_neighbors=focus, periods=periods,
            truth=focus_truth, true_objective_differences=[t['objective']-focus_truth[0]['objective']
                                                       for t in focus_truth[1:]],
            prediction_snapshots=[dict(paid_prefix=t, **{key:summary[key][t] for key in
                ('mean','difference_mean','difference_variance','probability_improving',
                 'identity_weights','f_only_identity_difference')}) for t in (0,10,11,12)],
            observed_raw_f=[row['raw'][0] for row in rows],
            observation_predictive_mean=summary['observation_predictive_mean'],
            difference_observation_covariance=summary['difference_observation_covariance'],
            interpretation='Ordered within-node then node-weight telescope; f/g identity terms average both evidence orders. Diagnostic on saved actual history, not an optimizer intervention.'))
    print(seed, 'done', cost['cpu_seconds'], flush=True)


def summarize():
    plan = read(MANIFEST)
    output = ROOT/plan['output_dir']
    cases, projections, attempts = [], [], []
    for seed in plan['focus_neighbors']:
        directory = output/f'seed_{seed}'
        cases.append(read(directory/'diagnosis.json'))
        projections.append(read(directory/'projection.json'))
    for directory in [output]+[output/f'seed_{s}' for s in plan['focus_neighbors']]:
        attempts.extend(json.loads(line) for line in (directory/'attempts.jsonl').read_text().splitlines())
    support = read(output/'source_support.json')
    function_support = read(output/'target_function_support.json')
    costs = {key:sum(r.get(key, 0) for r in attempts) for key in (
        'cpu_seconds','wall_seconds','new_source_calls','new_target_initial_calls',
        'new_target_search_calls','new_verification_calls','new_source_fits',
        'download_bytes','source_model_restorations','saved_raw_rows_replayed')}
    costs['new_truth_function_calls'] = sum(r.get('new_truth_function_calls',
                                                r.get('new_truth_calls', 0)) for r in attempts)
    completion = dict(status='complete', attempts=len(attempts),
        failed_attempts=[r for r in attempts if r['status'] != 'completed'], cost=costs,
        saved_query_prefixes_compared=6,
        saved_score_max_absolute_error=max(v for p in projections for r in p['parity'] for v in r['errors'].values()),
        attribution_max_absolute_error=max(s['reconstruction_error'] for p in projections for s in p['summary']['steps']),
        difference_covariance_max_absolute_error=max(p['difference_covariance_parity'] or 0 for p in projections),
        source_design_max_absolute_error=max(p['maximum_source_design_error'] for p in function_support['source_projection_parity']),
        roundoff_events=sum(len(p['roundoff_events']) for p in projections))
    save(output/'completion.json', completion)
    save(output/'summary.json', dict(cases=cases,
        source_pair_summary=[dict(family=f['family'], **f['ensemble']['posterior_common']['summary'])
                             for f in support['source_families']], completion=completion))
    plan['status'] = 'complete'
    plan['entrypoints'] = ['performance/diagnose_mainline_objective_attribution.py',
                          'performance/diagnose_mainline_objective_source_support.py']
    plan['additional_support_diagnostic'] = ('Compare actual source/query joint20 and learned f designs; '
        'source reconstruction detects an inconsistent provider or scale. Distances are descriptive, '
        'not proof of extrapolation or a selected model change.')
    save(MANIFEST, plan)
    print(completion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    run_case(read(MANIFEST), args.seed)
