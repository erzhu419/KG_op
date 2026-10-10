"""Keep the actual preceding recommendation in the existing terminal class."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.replay_legacy_mixture_budget import terminal_at_prefix
from performance.run_legacy_joint_population_independent_pilot import restore_source, target_problem
from performance.run_legacy_joint_population_matched_baselines import paired_fields
from performance.run_legacy_related_inventory_source import save
from performance.score_legacy_related_inventory_source import safe_terminal_values
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture
from representation.source_kernel_population import predict_kernel_population


def run():
    started = time.process_time()
    spec = json.loads((ROOT/'performance/manifests/legacy_joint_population_actual_retention_attribution_v1.json').read_text())
    inputs = json.loads((ROOT/spec['inputs']).read_text())
    gp = json.loads((ROOT/spec['kernel_population_fit']).read_text())
    previous = json.loads((ROOT/spec['previous_decisions']).read_text())
    _, priors = restore_source(inputs['target_plan'], inputs['source_spec'])
    bank = [tuple(x) for x in inputs['bank']]
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    jobs, rows = [], []
    for target in inputs['targets']:
        provider = no_outcomes(target_problem(inputs['target_plan'], target['config']))
        psi = (np.asarray(target['source_features'])*np.asarray(inputs['source_feature_scale'])
               +np.asarray(inputs['source_feature_center']))
        components = [(JointFunctionRiskPosterior([priors[n][0]], [priors[n][3]], psi),
                       JointFunctionRiskPosterior([priors[n][1]], [priors[n][2]], psi)) for n in spec['arms'][:2]]
        components.append(paired_fields([predict_kernel_population(gp['source_features'], target['source_features'], h)
                                         for h in gp['heads']]))
        model = JointResponseMixture(spec['arms'], components)
        observed = target['initialization']['paired_observations']
        for i, v in zip(target['initialization']['candidate_indices'], observed):
            model.update_raw(i, [v[0], v[1]-provider.tau])
        early = next(r for r in previous['rows'] if r['task_id'] == target['task_id']
                      and r['arm'] == spec['arms'][0] and r['paid_prefix'] == 2)
        terminal, _ = terminal_at_prefix(provider, bank[:2], {'paid_observations': observed[:2]},
                                         2, inputs['source_spec'], len(bank))
        terminal.retain_recommendation(early['retained_candidate'])
        terminal.retain_recommendation(early['action']['primary'])
        rewards, costs, actions = model.terminal_components(terminal, 358,
                                spec['terminal_draw_power'], spec['integration_seed'])
        for i, name in enumerate(spec['arms']):
            best = int(np.lexsort((costs[i], -rewards[i]))[0])
            rows.append({'task_id': target['task_id'], 'arm': name, 'paid_prefix': 4,
                'reference_retained_candidate': early['retained_candidate'],
                'actual_previous_recommendation': early['action']['primary'],
                'action_index': best, 'action': actions[best],
                'predicted_reward': float(rewards[i, best]),
                'predicted_verification_calls': float(costs[i, best])})
        jobs.append(terminal)
        print('actual-retention recommendations prepared', target['task_id'],
              [r['action']['primary'] for r in rows[-3:]], flush=True)
    save(out/'recommendations.json', {'execution': spec, 'rows': rows})
    old = json.loads((ROOT/'paper_artifacts/legacy_joint_population_independent_pilot_v1/results.json').read_text())
    for target, fold, terminal in zip(inputs['targets'], old['folds'], jobs):
        truth = np.asarray(fold['scoring_truth'])
        feasible = np.isin(np.arange(len(bank)), fold['feasible_policies'])
        safe, costs = safe_terminal_values(terminal, truth, feasible, 358)
        _, _, actions = terminal.evaluate(truth[None], 358)
        for row in [r for r in rows if r['task_id'] == target['task_id']]:
            a = row['action_index']
            if row['action'] != actions[a]:
                raise ValueError('frozen retention action differs from the scoring class')
            row['scores'] = {'conditional_safe_reward': float(safe[a]),
                'conditional_total_calls': float(4+costs[a]),
                'primary_feasible': bool(feasible[row['action']['primary']])}
    draws = np.random.default_rng(spec['bootstrap_seed']).integers(len(jobs),
                    size=(spec['bootstrap_draws'], len(jobs)))
    summaries, comparisons = {}, []
    for name in spec['arms']:
        selected = [r for r in rows if r['arm'] == name]
        summaries[name] = {'feasible_primary_count': sum(r['scores']['primary_feasible'] for r in selected),
            **{m: float(np.mean([r['scores'][m] for r in selected])) for m in
                ['conditional_safe_reward', 'conditional_total_calls']}}
        metrics = {}
        for m in ['conditional_safe_reward', 'conditional_total_calls']:
            diffs = np.asarray([r['scores'][m]-next(p for p in previous['rows']
                if p['task_id'] == r['task_id'] and p['arm'] == name and p['paid_prefix'] == 4)['scores'][m]
                for r in selected])
            metrics[m] = {'task_differences': diffs.tolist(), 'mean_difference': float(diffs.mean()),
                         'task_bootstrap_95_interval': np.quantile(diffs[draws].mean(axis=1), [.025, .975]).tolist()}
        comparisons.append({'arm': name, 'comparison': 'actual previous recommendation retention minus reference-only retention',
                            'metrics': metrics})
    result = {'execution': spec, 'rows': rows, 'summary': summaries, 'paired_comparisons': comparisons,
              'cost': {'new_source_calls': 0, 'new_target_calls': 0, 'new_verification_calls': 0,
                       'source_optimizations': 0, 'cached_logical_initialization_calls': 4*len(rows)},
              'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)
    print(json.dumps({'summary': summaries, 'cpu_seconds': result['cpu_seconds']}), flush=True)


if __name__ == '__main__':
    run()
