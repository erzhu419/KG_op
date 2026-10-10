"""Certify all final candidate posteriors and within-identity source evidence."""

from performance.complete_historical_source_heads import ROOT, read, save
from performance.certify_historical_final_risk_evidence import run as evidence_run
from performance.certify_historical_reused_risk_candidate import run as direct_run
from performance.condition_historical_reused_risk_candidate import run as condition_run

PLAN = 'performance/manifests/historical_final_source_certification_v1.json'


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists() and read(str(path.relative_to(ROOT))).get('all_registered_cases_complete'):
        return read(str(path.relative_to(ROOT)))
    rows = []
    for item in spec['cases']:
        candidate = read(item['candidate']); evidence = evidence_run(item['evidence_plan'])
        direct = direct_run(item['direct_plan'])
        reused_parent = evidence['unchanged_source_law']
        if item.get('completed_response_conditioning'):
            post_path = item['completed_response_conditioning']; post = read(post_path)
            passed = post['final_posterior_and_retention_certified']; method = 'completed final-law conditioning'
        elif reused_parent:
            record = next(r for r in read(spec['parent_precision'])['rows']
                if all(r[k] == candidate[k] for k in ('layout', 'selector', 'support', 'treatment', 'head')))
            passed = record['positive_mixture_precision_passed']; post = direct
            post_path = spec['parent_precision']; method = 'unchanged-law parent certificate'
        elif direct['posterior_functional_precision_passed']:
            post = direct; passed = True; post_path = read(item['direct_plan'])['output'] + '/results.json'
            method = 'all retained components directly certified;no further reduction'
        else:
            post = condition_run(item['conditioning_plan']); passed = post['final_posterior_and_retention_certified']
            post_path = read(item['conditioning_plan'])['output'] + '/results.json'; method = 'fresh fixed-candidate conditioning'
        row = {k: candidate[k] for k in ('layout', 'selector', 'support', 'treatment', 'head')}
        row.update(final_risk_scales=candidate['final_risk_scales'], fixed_response_scales=candidate['fixed_response_scales'],
            conditional_risk_stationarity_passed=candidate['conditional_risk_stationarity_passed'],
            candidate_results=item['candidate'], evidence_results=read(item['evidence_plan'])['output'] + '/results.json',
            source_evidence_precision_passed=evidence['source_evidence']['precision_gate_passed'],
            final_vs_parent_evidence_certified=evidence['final_vs_parent_evidence_certified'],
            posterior_functional_and_retention_certified=passed, posterior_results=post_path,
            posterior_method=method, candidate_component_file=post['candidate_component_file'],
            response_stationarity_at_final_risk=post['response_stationarity_at_final_risk'],
            final_candidate_numerically_certified=bool(passed and evidence['final_vs_parent_evidence_certified']))
        rows.append(row)
        phase = {'execution_plan': plan, 'rows': rows, 'registered_cases': len(spec['cases']),
            'all_registered_cases_complete': len(rows) == len(spec['cases']),
            'conditional_risk_stationarity_cases_passed': sum(r['conditional_risk_stationarity_passed'] for r in rows),
            'source_evidence_precision_cases_passed': sum(r['source_evidence_precision_passed'] for r in rows),
            'final_vs_parent_evidence_cases_certified': sum(r['final_vs_parent_evidence_certified'] for r in rows),
            'final_posterior_cases_certified': sum(r['posterior_functional_and_retention_certified'] for r in rows),
            'final_source_candidates_numerically_certified': sum(r['final_candidate_numerically_certified'] for r in rows),
            'source_registry_replaced': False, 'joint_source_learning_converged': False,
            'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0,
            'independent_confirmation': False}
        save(path, phase)
        print('final source batch', len(rows), '/', len(spec['cases']), 'numerically certified',
            phase['final_source_candidates_numerically_certified'], flush=True)
    return phase


if __name__ == '__main__':
    run()
