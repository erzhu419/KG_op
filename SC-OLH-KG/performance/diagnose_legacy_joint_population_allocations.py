"""Small allocation panel on cached observations; no new simulations."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.budgeted_guarded_kg import BudgetedTerminalActions
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_joint_population_independent_pilot import restore_source, target_problem
from performance.run_legacy_related_inventory_source import save
from performance.score_legacy_related_inventory_source import safe_terminal_values
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_observable_prior import legacy_exposure_descriptors


def run():
    started = time.process_time()
    spec = json.loads((ROOT/'performance/manifests/legacy_joint_population_independent_allocation_diagnosis_v1.json').read_text())
    folder = ROOT/spec['pilot']
    results = json.loads((folder/'results.json').read_text())
    inputs = json.loads((folder/'frozen_inputs.json').read_text())
    frozen = json.loads((folder/'recommendations.json').read_text())
    plan, execution = results['plan'], results['execution']
    source_spec = json.loads((ROOT/execution['legacy_replay']).read_text())['source_fit_contract']
    coordinate, arms = restore_source(plan, source_spec)
    bank = [tuple(x) for x in inputs['bank']]
    count = execution['adaptation_calls']; remaining = execution['total_target_call_cap']-count
    panels = []
    for config, observations, fold in zip(inputs['tasks'], frozen['initializations'], results['folds']):
        provider = no_outcomes(target_problem(plan, config))
        psi = coordinate.transform(legacy_exposure_descriptors(provider, bank, source_spec['dynamics_grid_dimension']))
        components = [(JointFunctionRiskPosterior([arms[n][0]], [arms[n][3]], psi),
                       JointFunctionRiskPosterior([arms[n][1]], [arms[n][2]], psi)) for n in spec['arms']]
        model = JointResponseMixture(spec['arms'], components)
        for index, value in zip(observations['candidate_indices'], observations['paired_observations']):
            model.update_raw(index, [value[0], value[1]-provider.tau])
        truth = np.asarray(fold['scoring_truth'])
        feasible = np.isin(np.arange(len(bank)), fold['feasible_policies'])
        protocol = {'objective_reference': float(np.dot(provider.weights, provider.ref_point)),
            'familywise_delta': source_spec['familywise_delta'], 'chance_alpha': provider.alpha,
            'comparison_budget': source_spec['comparison_budget'], 'comparison_delta': source_spec['comparison_delta']}
        for name, (f, g) in zip(spec['arms'], components):
            selected = next(r for r in frozen['rows'] if r['task_id'] == config['task_id'] and r['arm'] == name)['action']
            panel = BudgetedTerminalActions([selected['shortlist']], [selected['incumbent_position']], protocol)
            safe, costs = safe_terminal_values(panel, truth, feasible, remaining)
            integrations = []
            for seed in spec['integration_seeds']:
                reward, cost, actions = panel.evaluate_posterior(f, g, remaining, spec['integration_power'], seed)
                best = int(np.lexsort((cost, -reward))[0])
                integrations.append({'seed': seed, 'reward': reward.tolist(),
                    'expected_total_calls': (cost+count).tolist(), 'best_panel_action_index': best})
            panels.append({'task_id': config['task_id'], 'arm': name, 'frozen_action': selected,
                'actions': actions, 'conditional_safe_rewards': safe.tolist(),
                'conditional_total_calls': (costs+count).tolist(), 'integrations': integrations})
        print('allocation panel complete', config['task_id'], flush=True)
    result = {'protocol': spec, 'panels': panels, 'new_simulator_calls': 0, 'source_optimizations': 0,
              'cpu_seconds': time.process_time()-started}
    save(ROOT/spec['output'], result)
    print(json.dumps({'panels': len(panels), 'cpu_seconds': result['cpu_seconds']}), flush=True)


if __name__ == '__main__':
    run()
