"""Opened-truth headroom and verifier curves for already frozen actions."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.guarded_deployment_kg import conditional_safety_power
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.replay_legacy_mixture_budget import terminal_at_prefix
from performance.run_legacy_joint_population_independent_pilot import target_problem
from performance.run_legacy_related_inventory_source import save
from performance.score_legacy_related_inventory_source import safe_terminal_values


def run():
    started = time.process_time()
    spec = json.loads((ROOT/'performance/manifests/legacy_joint_population_public_mean_decision_v1.json').read_text())
    inputs = json.loads((ROOT/spec['inputs']).read_text())
    result = json.loads((ROOT/spec['output']/'results.json').read_text())
    scoring = json.loads((ROOT/spec['target_scoring']).read_text())
    bank = [tuple(p) for p in inputs['bank']]; rows = []
    for target in inputs['targets']:
        provider = no_outcomes(target_problem(inputs['target_plan'], target['config']))
        terminal, _ = terminal_at_prefix(provider, bank[:2],
            {'paid_observations': target['initialization']['paired_observations']}, 2,
            inputs['source_spec'], len(bank))
        terminal.retain_recommendation(spec['public_retained_candidate'])
        fold = next(f for f in scoring['folds'] if f['task_id'] == target['task_id'])
        truth = np.asarray(fold['scoring_truth'])  # [f, eta_f, g, eta_g]
        feasible = np.isin(np.arange(len(bank)), fold['feasible_policies'])
        quality = np.clip(1-truth[:, 0]/terminal.protocol['objective_reference'], 0, 1)*feasible
        safe, calls = safe_terminal_values(terminal, truth, feasible, 360)
        _, _, actions = terminal.evaluate(truth[None], 360)
        best = int(np.lexsort((calls, -safe))[0])
        chosen = [r for r in result['rows'] if r['task_id'] == target['task_id']]
        main = next(r for r in chosen if r['arm'] == spec['main_arm'])
        curves = {}
        for count in (60, 120, 180, 240, 360):
            power = conditional_safety_power(truth[:, 2], np.exp(truth[:, 3]/2), count,
                terminal.protocol['chance_alpha'], terminal.protocol['familywise_delta'])
            curves[str(count)] = {r['arm']: float(quality[r['action']['primary']]*power[r['action']['primary']])
                for r in chosen}  # All 42 frozen actions are direct verification.
        rows.append({'task_id': target['task_id'], 'best_same_class_action': actions[best],
            'best_same_class_reward': float(safe[best]), 'perfect_certification_quality_bound': float(quality.max()),
            'main_reward': main['scores']['conditional_safe_reward'],
            'main_loss_to_best_same_class': float(safe[best]-main['scores']['conditional_safe_reward']),
            'main_loss_to_perfect_certification_bound': float(quality.max()-main['scores']['conditional_safe_reward']),
            'fixed_primary_direct_verification_curves': curves})
    summary = {k: float(np.mean([r[k] for r in rows])) for k in
        ('best_same_class_reward', 'perfect_certification_quality_bound', 'main_reward',
         'main_loss_to_best_same_class', 'main_loss_to_perfect_certification_bound')}
    summary['fixed_primary_direct_verification_curves'] = {str(n): {a: float(np.mean([
        r['fixed_primary_direct_verification_curves'][str(n)][a] for r in rows]))
        for a in spec['arms']} for n in (60, 120, 180, 240, 360)}
    diagnostic = {'scope': 'opened-truth diagnosis for frozen recommendations; no budget-specific reranking, KG selection or new certificate',
        'rows': rows, 'summary': summary,
        'cost': {'new_source_fits': 0, 'new_simulator_calls': 0, 'new_verification_calls': 0,
                 'new_posterior_terminal_integrations': 0}, 'cpu_seconds': time.process_time()-started}
    save(ROOT/spec['output']/'ceiling_diagnostic.json', diagnostic)
    print(json.dumps({'summary': summary, 'cpu_seconds': diagnostic['cpu_seconds']}), flush=True)


if __name__ == '__main__':
    run()
