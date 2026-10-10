"""Freeze two source assignments from an outcome-free public function design."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_joint_population_matched_baselines import standardize
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates


def coverage_sites(catalog, anchors, count):
    """Pivot the whitened public function span, then reduce mean OLS variance."""
    u, singular, _ = np.linalg.svd(catalog, full_matrices=False)
    rank = np.linalg.matrix_rank(catalog)
    z = u[:, :rank]*np.sqrt(len(catalog))
    selected = list(anchors)
    q, _ = np.linalg.qr(z[selected].T, mode='reduced')
    while len(selected) < rank:
        residual = z-(z@q)@q.T
        gain = np.sum(residual**2, axis=1); gain[selected] = -np.inf
        j = int(np.argmax(gain)); selected.append(j)
        direction = residual[j]/np.sqrt(gain[j])
        q = np.column_stack([q, direction])
    while len(selected) < count:
        a = 6*z[anchors].T@z[anchors]+z[selected[len(anchors):]].T@z[selected[len(anchors):]]
        inverse = np.linalg.inv(a)
        # Sherman-Morrison reduction in average prediction variance over all public policies.
        solved = z@inverse
        gain = np.mean((solved@z.T)**2, axis=0)/(1+np.sum(solved*z, axis=1))
        gain[selected] = -np.inf; selected.append(int(np.argmax(gain)))
    return selected[len(anchors):]


def geometry(raw, reference, sites):
    selected, center, scale = standardize(np.vstack([r[s] for r, s in zip(raw, sites)]))
    catalog = (reference-center)/scale; basis = OrthogonalLowOrderBasis(catalog)
    x = np.column_stack([np.ones(len(selected)), basis.transform(selected)])
    q = np.column_stack([np.ones(len(catalog)), basis.transform(catalog)])
    weights = np.linalg.lstsq(x.T, q.T, rcond=None)[0].T
    return {'actual_mean_design_rank': int(np.linalg.matrix_rank(x)),
        'coefficient_count': x.shape[1], 'actual_mean_design_condition_number': float(np.linalg.cond(x)),
        'public_catalog_ols_prediction_leverage_quantiles': np.quantile(np.sum(weights**2, axis=1), [0, .5, .9, 1]).tolist()}


def run():
    parent = read('performance/manifests/legacy_joint_population_source_layout_robustness_execution_v1.json')
    base = read(parent['reference_source_execution'])
    public = read(Path(base['source_data']).with_name('frozen_inputs.json'))
    bank = [tuple(p) for p in public['bank']]
    raw = [public_control_mean_coordinates(no_outcomes(problem_for(public['plan'], public['tasks'][i])), bank)
           for i in base['training_tasks']]
    psi, _, _ = standardize(raw[0]); basis = OrthogonalLowOrderBasis(psi)
    catalog = np.column_stack([np.ones(len(psi)), basis.transform(psi)])
    anchors = base['source_site_indices_by_task'][0][:4]
    extras = coverage_sites(catalog, anchors, 28)
    first = [anchors+extras[i::6] for i in range(6)]
    layouts = {'3': first, '4': first[::-1]}
    plan = {**parent, 'status': 'registered outcome-free source-coverage repair; source fits not executed',
        'layouts': layouts,
        'selection': 'Read only the frozen public bank/task configurations. Build the unchanged26-function public mean dictionary with public-catalog normalization for design only; whiten its full joint span for invariant design selection, without changing the fitted prior basis. Retain four shared anchors. Greedily add23 policies by largest orthogonal residual norm to span all27 coefficients, then add one by largest reduction of uniform-public-catalog OLS prediction variance. No diagnostic candidate list, source response, holdout or target observation enters selection. Assign24 extras round-robin to six source tasks; layout4 reverses task assignment, preserving policy coverage and144 calls.',
        'public_geometry': {name: geometry(raw, raw[0], sites) for name, sites in layouts.items()},
        'public_function_pivot_order': extras,
        'source_gate': 'Evaluate both predeclared assignments and retain both. Before any target probe, require structural g MSE to decrease and joint/g three-repeat predictive log densities to improve on both opened source holdouts against fixed layout0, separately in each assignment. A failing assignment retains the negative result and does not proceed to target integration. No selecting a better assignment, changing identity treatments or pooling labels.',
        'next_action': 'Fit and freeze the184 native source objectives and12 source forecasts with the unchanged Stage39 treatments and fitting rules. Score source predictions first. Only passing assignments may enter a separately frozen eight-direct-action target diagnostic; no automatic73-policy ranking or KG matrix.',
        'interpretation': 'opened-source design repair at the same144 calls; two source task assignments with the same28-policy coverage are development contrasts, not independent source populations or a demonstrated method advantage',
        'output': 'paper_artifacts/legacy_joint_population_function_coverage_source_v1'}
    if not all(g['actual_mean_design_rank'] == g['coefficient_count'] for g in plan['public_geometry'].values()):
        raise ValueError('proposed actual mean dictionary is not spanned; do not launch source fits')
    save(ROOT/'performance/manifests/legacy_joint_population_function_coverage_source_plan_v1.json', plan)
    print({'layouts': layouts, 'public_geometry': plan['public_geometry']}, flush=True)


if __name__ == '__main__':
    run()
